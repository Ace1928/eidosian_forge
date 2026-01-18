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
def is_string_attribute(cls, key_type, key_name, comparator):
    if key_type == cls._ATTRIBUTE_IDENTIFIER and key_name not in cls.NUMERIC_ATTRIBUTES:
        if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
            raise MlflowException(f"Invalid comparator '{comparator}' not one of '{cls.VALID_STRING_ATTRIBUTE_COMPARATORS}'")
        return True
    return False