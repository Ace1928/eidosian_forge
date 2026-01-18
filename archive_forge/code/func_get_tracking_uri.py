import logging
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Union
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
def get_tracking_uri() -> str:
    """Get the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.

    Returns:
        The tracking URI.

    .. code-block:: python

        import mlflow

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

    .. code-block:: text

        Current tracking uri: file:///.../mlruns
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif (uri := MLFLOW_TRACKING_URI.get()):
        return uri
    else:
        return path_to_local_file_uri(os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH))