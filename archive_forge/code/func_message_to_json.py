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
def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    json_dict_with_int64_as_str = json.loads(MessageToJson(message, preserving_proto_field_name=True))
    json_dict_with_int64_fields_only = _mark_int64_fields(message)
    json_dict_with_int64_as_numbers = _merge_json_dicts(json_dict_with_int64_fields_only, json_dict_with_int64_as_str)
    return json.dumps(json_dict_with_int64_as_numbers, indent=2)