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
def mysql_comparison_func(column, value):
    if not isinstance(column.type, sa.types.String):
        return comparison_func(column, value)
    templates = {'=': '({column} = :value AND BINARY {column} = :value)', '!=': '({column} != :value OR BINARY {column} != :value)', 'LIKE': '({column} LIKE :value AND BINARY {column} LIKE :value)'}
    if comparator in templates:
        column = f'{column.class_.__tablename__}.{column.key}'
        return sa.text(templates[comparator].format(column=column)).bindparams(sa.bindparam('value', value=value, unique=True))
    return comparison_func(column, value)