import os
import yaml
from mlflow.exceptions import ExecutionException
from mlflow.projects import env_type
from mlflow.tracking import artifact_utils
from mlflow.utils import data_utils
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME
from mlflow.utils.file_utils import get_local_path_or_none
from mlflow.utils.string_utils import is_string_type, quote
def compute_parameters(self, user_parameters, storage_dir):
    """
        Given a dict mapping user-specified param names to values, computes parameters to
        substitute into the command for this entry point. Returns a tuple (params, extra_params)
        where `params` contains key-value pairs for parameters specified in the entry point
        definition, and `extra_params` contains key-value pairs for additional parameters passed
        by the user.

        Note that resolving parameter values can be a heavy operation, e.g. if a remote URI is
        passed for a parameter of type `path`, we download the URI to a local path within
        `storage_dir` and substitute in the local path as the parameter value.

        If `storage_dir` is `None`, report path will be return as parameter.
        """
    if user_parameters is None:
        user_parameters = {}
    self._validate_parameters(user_parameters)
    final_params = {}
    extra_params = {}
    parameter_keys = list(self.parameters.keys())
    for key in parameter_keys:
        param_obj = self.parameters[key]
        key_position = parameter_keys.index(key)
        value = user_parameters[key] if key in user_parameters else self.parameters[key].default
        final_params[key] = param_obj.compute_value(value, storage_dir, key_position)
    for key in user_parameters:
        if key not in final_params:
            extra_params[key] = user_parameters[key]
    return (self._sanitize_param_dict(final_params), self._sanitize_param_dict(extra_params))