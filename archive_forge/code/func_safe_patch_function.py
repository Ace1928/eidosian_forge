import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
def safe_patch_function(*args, **kwargs):
    """
        A safe wrapper around the specified `patch_function` implementation designed to
        handle exceptions thrown during the execution of `patch_function`. This wrapper
        distinguishes exceptions thrown from the underlying / original function
        (`<destination>.<function_name>`) from exceptions thrown from other parts of
        `patch_function`. This distinction is made by passing an augmented version of the
        underlying / original function to `patch_function` that uses nonlocal state to track
        whether or not it has been executed and whether or not it threw an exception.
        Exceptions thrown from the underlying / original function are propagated to the caller,
        while exceptions thrown from other parts of `patch_function` are caught and logged as
        warnings.
        """
    is_silent_mode = get_autologging_config(autologging_integration, 'silent', False)
    with set_mlflow_events_and_warnings_behavior_globally(reroute_warnings=True, disable_event_logs=is_silent_mode, disable_warnings=is_silent_mode), set_non_mlflow_warnings_behavior_for_current_thread(reroute_warnings=True, disable_warnings=is_silent_mode):
        if is_testing():
            preexisting_run_for_testing = mlflow.active_run()
        exclusive = get_autologging_config(autologging_integration, 'exclusive', False)
        user_created_fluent_run_is_active = mlflow.active_run() and (not _AutologgingSessionManager.active_session())
        active_session_failed = _AutologgingSessionManager.active_session() is not None and _AutologgingSessionManager.active_session().state == 'failed'
        if active_session_failed or autologging_is_disabled(autologging_integration) or (user_created_fluent_run_is_active and exclusive) or mlflow.utils.autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED:
            with set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings=False, reroute_warnings=False):
                return original(*args, **kwargs)
        original_has_been_called = False
        original_result = None
        failed_during_original = False
        patch_function_run_for_testing = None
        patch_function_exception = None

        def try_log_autologging_event(log_fn, *args):
            try:
                log_fn(*args)
            except Exception as e:
                _logger.debug("Failed to log autologging event via '%s'. Exception: %s", log_fn, e)

        def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
            try:
                try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_start, session, destination, function_name, og_args, og_kwargs)
                original_fn_result = original_fn(*og_args, **og_kwargs)
                try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_success, session, destination, function_name, og_args, og_kwargs)
                return original_fn_result
            except Exception as original_fn_e:
                try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_error, session, destination, function_name, og_args, og_kwargs, original_fn_e)
                nonlocal failed_during_original
                failed_during_original = True
                raise
        with _AutologgingSessionManager.start_session(autologging_integration) as session:
            try:

                def call_original(*og_args, **og_kwargs):

                    def _original_fn(*_og_args, **_og_kwargs):
                        if is_testing():
                            _validate_args(autologging_integration, function_name, args, kwargs, og_args, og_kwargs)
                            nonlocal patch_function_run_for_testing
                            patch_function_run_for_testing = mlflow.active_run()
                        nonlocal original_has_been_called
                        original_has_been_called = True
                        nonlocal original_result
                        with set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings=False, reroute_warnings=False):
                            original_result = original(*_og_args, **_og_kwargs)
                            return original_result
                    return call_original_fn_with_event_logging(_original_fn, og_args, og_kwargs)
                call_original = update_wrapper_extended(call_original, original)
                try_log_autologging_event(AutologgingEventLogger.get_logger().log_patch_function_start, session, destination, function_name, args, kwargs)
                if patch_is_class:
                    patch_function.call(call_original, *args, **kwargs)
                else:
                    patch_function(call_original, *args, **kwargs)
                session.state = 'succeeded'
                try_log_autologging_event(AutologgingEventLogger.get_logger().log_patch_function_success, session, destination, function_name, args, kwargs)
            except Exception as e:
                session.state = 'failed'
                patch_function_exception = e
                if failed_during_original or is_testing():
                    raise
            if is_testing() and (not preexisting_run_for_testing):
                assert not mlflow.active_run(), f'Autologging integration {autologging_integration} leaked an active run'
                if patch_function_run_for_testing:
                    _validate_autologging_run(autologging_integration, patch_function_run_for_testing.info.run_id)
            try:
                if original_has_been_called:
                    return original_result
                else:
                    return call_original_fn_with_event_logging(original, args, kwargs)
            finally:
                if patch_function_exception is not None and (not failed_during_original):
                    try_log_autologging_event(AutologgingEventLogger.get_logger().log_patch_function_error, session, destination, function_name, args, kwargs, patch_function_exception)
                    _logger.warning('Encountered unexpected error during %s autologging: %s', autologging_integration, patch_function_exception)