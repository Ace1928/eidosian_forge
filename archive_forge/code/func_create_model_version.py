import logging
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names
def create_model_version(self, name, source, run_id=None, tags=None, run_link=None, description=None, await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS, local_model_path=None):
    """Create a new model version from given source.

        Args:
            name: Name of the containing registered model.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            await_creation_for: Number of seconds to wait for the model version to finish being
                created and is in ``READY`` status. By default, the function
                waits for five minutes. Specify 0 or None to skip waiting.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
            backend.

        """
    tags = tags if tags else {}
    tags = [ModelVersionTag(key, str(value)) for key, value in tags.items()]
    arg_names = _get_arg_names(self.store.create_model_version)
    if 'local_model_path' in arg_names:
        mv = self.store.create_model_version(name, source, run_id, tags, run_link, description, local_model_path=local_model_path)
    else:
        mv = self.store.create_model_version(name, source, run_id, tags, run_link, description)
    if await_creation_for and await_creation_for > 0:
        self.store._await_model_version_creation(mv, await_creation_for)
    return mv