import base64
import json
import requests
from mlflow.environment_variables import (
from mlflow.exceptions import (
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.request_utils import (
from mlflow.utils.string_utils import strip_suffix
def extract_api_info_for_service(service, path_prefix):
    """Return a dictionary mapping each API method to a tuple (path, HTTP method)"""
    service_methods = service.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = _get_path(path_prefix, endpoint.path)
        res[service().GetRequestClass(service_method)] = (endpoint_path, endpoint.method)
    return res