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
def set_registry_uri(uri: str) -> None:
    """Set the registry server URI. This method is especially useful if you have a registry server
    that's different from the tracking server.

    Args:
        uri: An empty string, or a local file path, prefixed with ``file:/``. Data is stored
            locally at the provided file (or ``./mlruns`` if empty). An HTTP URI like
            ``https://my-tracking-server:5000``. A Databricks workspace, provided as the string
            "databricks" or, to use a Databricks CLI
            `profile <https://github.com/databricks/databricks-cli#installation>`_,
            "databricks://<profileName>".

    .. code-block:: python
        :caption: Example

        import mflow

        # Set model registry uri, fetch the set uri, and compare
        # it with the tracking uri. They should be different
        mlflow.set_registry_uri("sqlite:////tmp/registry.db")
        mr_uri = mlflow.get_registry_uri()
        print(f"Current registry uri: {mr_uri}")
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

        # They should be different
        assert tracking_uri != mr_uri

    .. code-block:: text
        :caption: Output

        Current registry uri: sqlite:////tmp/registry.db
        Current tracking uri: file:///.../mlruns

    """
    global _registry_uri
    _registry_uri = uri