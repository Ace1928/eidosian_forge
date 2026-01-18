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
def get_databricks_profile_uri_from_artifact_uri(uri, result_scheme='databricks'):
    """
    Retrieves the netloc portion of the URI as a ``databricks://`` or `databricks-uc://` URI,
    if it is a proper Databricks profile specification, e.g.
    ``profile@databricks`` or ``secret_scope:key_prefix@databricks``.
    """
    parsed = urllib.parse.urlparse(uri)
    if not parsed.netloc or parsed.hostname != result_scheme:
        return None
    if not parsed.username:
        return result_scheme
    validate_db_scope_prefix_info(parsed.username, parsed.password)
    key_prefix = ':' + parsed.password if parsed.password else ''
    return f'{result_scheme}://' + parsed.username + key_prefix