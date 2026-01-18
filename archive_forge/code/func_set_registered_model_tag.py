import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def set_registered_model_tag(self, name, key, value):
    """Set a tag for the registered model.

        Args:
            name: Registered model name.
            key: Tag key to log.
            value: Tag value log.

        Returns:
            None
        """
    self.store.set_registered_model_tag(name, RegisteredModelTag(key, str(value)))