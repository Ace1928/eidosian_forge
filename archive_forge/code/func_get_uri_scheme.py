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
def get_uri_scheme(uri_or_path):
    scheme = urllib.parse.urlparse(uri_or_path).scheme
    if any((scheme.lower().startswith(db) for db in DATABASE_ENGINES)):
        return extract_db_type_from_uri(uri_or_path)
    return scheme