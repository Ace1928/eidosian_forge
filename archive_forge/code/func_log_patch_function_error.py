import warnings
from mlflow.utils.autologging_utils import _logger
def log_patch_function_error(self, session, patch_obj, function_name, call_args, call_kwargs, exception):
    """Called when execution of a patched API associated with an autologging integration
        (e.g., `sklearn.linear_model.LogisticRegression.fit()`) terminates with an exception.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the patched API was called.
            function_name: The name of the patched API that was called.
            call_args: The positional arguments passed to the patched API call.
            call_kwargs: The keyword arguments passed to the patched API call.
            exception: The exception that caused the patched API call to terminate.
        """
    _logger.debug("Patched API call '%s.%s' for %s autologging threw exception. Patched API was called with args '%s' and kwargs '%s'. Exception: %s", patch_obj, function_name, session.integration, call_args, call_kwargs, exception)