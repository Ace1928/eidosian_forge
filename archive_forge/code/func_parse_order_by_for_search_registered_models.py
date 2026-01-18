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
def parse_order_by_for_search_registered_models(cls, order_by):
    token_value, is_ascending = cls._parse_order_by_string(order_by)
    identifier = SearchExperimentsUtils._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_KEYS_REGISTERED_MODELS)
    return (identifier['type'], identifier['key'], is_ascending)