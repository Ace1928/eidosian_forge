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
def set_tracking_uri(uri: Union[str, Path]) -> None:
    """
    Set the tracking server URI. This does not affect the
    currently active run (if one exists), but takes effect for successive runs.

    Args:
        uri:

            - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
              locally at the provided file (or ``./mlruns`` if empty).
            - An HTTP URI like ``https://my-tracking-server:5000``.
            - A Databricks workspace, provided as the string "databricks" or, to use a Databricks
              CLI `profile <https://github.com/databricks/databricks-cli#installation>`_,
              "databricks://<profileName>".
            - A :py:class:`pathlib.Path` instance

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow

        mlflow.set_tracking_uri("file:///tmp/my_tracking")
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

    .. code-block:: text
        :caption: Output

        Current tracking uri: file:///tmp/my_tracking
    """
    if isinstance(uri, Path):
        uri = uri.absolute().resolve().as_uri()
    global _tracking_uri
    _tracking_uri = uri