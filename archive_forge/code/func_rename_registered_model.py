import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def rename_registered_model(self, name, new_name):
    """Update registered model name.

        Args:
            name: Name of the registered model to update.
            new_name: New proposed name for the registered model.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
    if new_name.strip() == '':
        raise MlflowException('The name must not be an empty string.')
    return self.store.rename_registered_model(name=name, new_name=new_name)