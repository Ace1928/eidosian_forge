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
def get_db_info_from_uri(uri):
    """
    Get the Databricks profile specified by the tracking URI (if any), otherwise
    returns None.
    """
    parsed_uri = urllib.parse.urlparse(uri)
    if parsed_uri.scheme == 'databricks' or parsed_uri.scheme == _DATABRICKS_UNITY_CATALOG_SCHEME:
        if parsed_uri.netloc == '':
            raise MlflowException(f"URI is formatted incorrectly: no netloc in URI '{uri}'. This may be the case if there is only one slash in the URI.")
        profile_tokens = parsed_uri.netloc.split(':')
        parsed_scope = profile_tokens[0]
        if len(profile_tokens) == 1:
            parsed_key_prefix = None
        elif len(profile_tokens) == 2:
            parsed_key_prefix = profile_tokens[1]
        else:
            parsed_key_prefix = ':'.join(profile_tokens[1:])
        validate_db_scope_prefix_info(parsed_scope, parsed_key_prefix)
        return (parsed_scope, parsed_key_prefix)
    return (None, None)