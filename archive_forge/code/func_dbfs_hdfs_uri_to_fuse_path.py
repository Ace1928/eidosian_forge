import os
import pathlib
import posixpath
import re
import urllib.parse
import uuid
from typing import Any, Tuple
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.os import is_windows
from mlflow.utils.validation import _validate_db_type_string
def dbfs_hdfs_uri_to_fuse_path(dbfs_uri):
    """Converts the provided DBFS URI into a DBFS FUSE path

    Args:
        dbfs_uri: A DBFS URI like "dbfs:/my-directory". Can also be a scheme-less URI like
            "/my-directory" if running in an environment where the default HDFS filesystem
            is "dbfs:/" (e.g. Databricks)

    Returns:
        A DBFS FUSE-style path, e.g. "/dbfs/my-directory"

    """
    if not is_valid_dbfs_uri(dbfs_uri) and dbfs_uri == posixpath.abspath(dbfs_uri):
        dbfs_uri = 'dbfs:' + dbfs_uri
    if not dbfs_uri.startswith(_DBFS_HDFS_URI_PREFIX):
        raise MlflowException(f"Path '{dbfs_uri}' did not start with expected DBFS URI prefix '{_DBFS_HDFS_URI_PREFIX}'")
    return _DBFS_FUSE_PREFIX + dbfs_uri[len(_DBFS_HDFS_URI_PREFIX):]