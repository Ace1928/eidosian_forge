import warnings
from mlflow.utils.autologging_utils import _logger
def log_autolog_called(self, integration, call_args, call_kwargs):
    """Called when the `autolog()` method for an autologging integration
        is invoked (e.g., when a user invokes `mlflow.sklearn.autolog()`)

        Args:
            integration: The autologging integration for which `autolog()` was called.
            call_args: **DEPRECATED** The positional arguments passed to the `autolog()` call.
                This field is empty in MLflow > 1.13.1; all arguments are passed in
                keyword form via `call_kwargs`.
            call_kwargs: The arguments passed to the `autolog()` call in keyword form.
                Any positional arguments should also be converted to keyword form
                and passed via `call_kwargs`.
        """
    if len(call_args) > 0:
        warnings.warn('Received %d positional arguments via `call_args`. `call_args` is deprecated in MLflow > 1.13.1, and all arguments should be passed in keyword form via `call_kwargs`.' % len(call_args), category=DeprecationWarning, stacklevel=2)
    _logger.debug("Called autolog() method for %s autologging with args '%s' and kwargs '%s'", integration, call_args, call_kwargs)