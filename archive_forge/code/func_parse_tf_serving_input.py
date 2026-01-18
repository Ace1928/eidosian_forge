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
def parse_tf_serving_input(inp_dict, schema=None):
    """
    Args:
        inp_dict: A dict deserialized from a JSON string formatted as described in TF's
            serving API doc
            (https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
        schema: MLflow schema used when parsing the data.
    """
    if 'signature_name' in inp_dict:
        raise MlflowInvalidInputException('"signature_name" parameter is currently not supported')
    if not (list(inp_dict.keys()) == ['instances'] or list(inp_dict.keys()) == ['inputs']):
        raise MlflowInvalidInputException(f'One of "instances" and "inputs" must be specified (not both or any other keys).Received: {list(inp_dict.keys())}')
    try:
        if 'instances' in inp_dict:
            return parse_instances_data(inp_dict, schema)
        else:
            return _cast_schema_type(inp_dict['inputs'], schema)
    except MlflowException as e:
        raise e
    except Exception as e:
        raise MlflowInvalidInputException(f'Ensure that the input is a valid JSON-formatted string.\nError: {e!r}') from e