import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def get_latest_versions(self, name, stages=None):
    """Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        Args:
            name: Name of the registered model from which to get the latest versions.
            stages: List of desired stages. If input list is None, return latest versions for
                'Staging' and 'Production' stages.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.

        """
    return self.store.get_latest_versions(name, stages)