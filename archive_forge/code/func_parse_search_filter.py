import ast
import base64
import json
import math
import operator
import re
import shlex
import sqlparse
from packaging.version import Version
from sqlparse.sql import (
from sqlparse.tokens import Token as TokenType
from mlflow.entities import RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.utils.mlflow_tags import (
@classmethod
def parse_search_filter(cls, filter_string):
    if not filter_string:
        return []
    try:
        parsed = sqlparse.parse(filter_string)
    except Exception:
        raise MlflowException(f"Error on parsing filter '{filter_string}'", error_code=INVALID_PARAMETER_VALUE)
    if len(parsed) == 0 or not isinstance(parsed[0], Statement):
        raise MlflowException(f"Invalid filter '{filter_string}'. Could not be parsed.", error_code=INVALID_PARAMETER_VALUE)
    elif len(parsed) > 1:
        raise MlflowException("Search filter contained multiple expression '%s'. Provide AND-ed expression list." % filter_string, error_code=INVALID_PARAMETER_VALUE)
    return cls._process_statement(parsed[0])