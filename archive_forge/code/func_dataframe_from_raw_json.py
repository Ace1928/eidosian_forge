import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Dict, Optional
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
def dataframe_from_raw_json(path_or_str, schema=None, pandas_orient: str='split'):
    """Parse raw json into a pandas.Dataframe.

    If schema is provided this methods will attempt to cast data types according to the schema. This
    include base64 decoding for binary columns.

    Args:
        path_or_str: Path to a json file or a json string.
        schema: MLflow schema used when parsing the data.
        pandas_orient: pandas data frame convention used to store the data.

    Returns:
        pandas.DataFrame.
    """
    if os.path.exists(path_or_str):
        with open(path_or_str) as f:
            parsed_json = json.load(f)
    else:
        parsed_json = json.loads(path_or_str)
    return dataframe_from_parsed_json(parsed_json, pandas_orient, schema)