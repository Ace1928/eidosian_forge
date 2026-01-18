import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def get_model_version_stages(self, name, version):
    """
        Returns:
            A list of valid stages.
        """
    return self.store.get_model_version_stages(name, version)