from functools import partial
from mlflow.environment_variables import MLFLOW_REGISTRY_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.databricks_workspace_model_registry_rest_store import (
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service.utils import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import (
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
def get_registry_uri() -> str:
    """Get the current registry URI. If none has been specified, defaults to the tracking URI.

    Returns:
        The registry URI.

    .. code-block:: python

        # Get the current model registry uri
        mr_uri = mlflow.get_registry_uri()
        print(f"Current model registry uri: {mr_uri}")

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        # They should be the same
        assert mr_uri == tracking_uri

    .. code-block:: text

        Current model registry uri: file:///.../mlruns
        Current tracking uri: file:///.../mlruns

    """
    return _get_registry_uri_from_context() or get_tracking_uri()