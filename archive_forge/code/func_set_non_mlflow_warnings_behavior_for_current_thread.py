import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
@contextmanager
def set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings, reroute_warnings):
    """
    Context manager that modifies the behavior of non-MLflow warnings upon entry, according to the
    specified parameters.

    Args:
        disable_warnings: If `True`, disable  (mutate & discard) non-MLflow warnings. If `False`,
            do not disable non-MLflow warnings.
        reroute_warnings: If `True`, reroute non-MLflow warnings to an MLflow event logger with
            level WARNING. If `False`, do not reroute non-MLflow warnings.
    """
    prev_disablement_state = _WARNINGS_CONTROLLER.get_warnings_disablement_state_for_current_thread()
    prev_rerouting_state = _WARNINGS_CONTROLLER.get_warnings_rerouting_state_for_current_thread()
    try:
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(disabled=disable_warnings)
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(rerouted=reroute_warnings)
        yield
    finally:
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(disabled=prev_disablement_state)
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(rerouted=prev_rerouting_state)