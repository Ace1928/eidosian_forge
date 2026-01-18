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
def append_to_uri_query_params(uri, *query_params: Tuple[str, Any]) -> str:
    """Appends the specified query parameters to an existing URI.

    Args:
        uri: The URI to which to append query parameters.
        query_params: Query parameters to append. Each parameter should
            be a 2-element tuple. For example, ``("key", "value")``.
    """
    parsed_uri = urllib.parse.urlparse(uri)
    parsed_query = urllib.parse.parse_qsl(parsed_uri.query)
    new_parsed_query = parsed_query + list(query_params)
    new_query = urllib.parse.urlencode(new_parsed_query)
    new_parsed_uri = parsed_uri._replace(query=new_query)
    return urllib.parse.urlunparse(new_parsed_uri)