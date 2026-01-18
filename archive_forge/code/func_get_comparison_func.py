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
@staticmethod
def get_comparison_func(comparator):
    return {'>': operator.gt, '>=': operator.ge, '=': operator.eq, '!=': operator.ne, '<=': operator.le, '<': operator.lt, 'LIKE': _like, 'ILIKE': _ilike, 'IN': lambda x, y: x in y, 'NOT IN': lambda x, y: x not in y}[comparator]