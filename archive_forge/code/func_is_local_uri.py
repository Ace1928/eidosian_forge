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
def is_local_uri(uri, is_tracking_or_registry_uri=True):
    """Returns true if the specified URI is a local file path (/foo or file:/foo).

    Args:
        uri: The URI.
        is_tracking_uri: Whether or not the specified URI is an MLflow Tracking or MLflow
            Model Registry URI. Examples of other URIs are MLflow artifact URIs,
            filesystem paths, etc.
    """
    if uri == 'databricks' and is_tracking_or_registry_uri:
        return False
    if is_windows() and uri.startswith('\\\\'):
        return False
    parsed_uri = urllib.parse.urlparse(uri)
    scheme = parsed_uri.scheme
    if scheme == '':
        return True
    is_remote_hostname = parsed_uri.hostname and (not (parsed_uri.hostname == '.' or parsed_uri.hostname.startswith('localhost') or parsed_uri.hostname.startswith('127.0.0.1')))
    if scheme == 'file':
        if is_remote_hostname:
            raise MlflowException(f'{uri} is not a valid remote uri. For remote access on windows, please consider using a different scheme such as SMB (e.g. smb://<hostname>/<path>).')
        return True
    if is_remote_hostname:
        return False
    if is_windows() and len(scheme) == 1 and (scheme.lower() == pathlib.Path(uri).drive.lower()[0]):
        return True
    return False