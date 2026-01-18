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
def call_endpoint(host_creds, endpoint, method, json_body, response_proto, extra_headers=None):
    if json_body:
        json_body = json.loads(json_body)
    call_kwargs = {'host_creds': host_creds, 'endpoint': endpoint, 'method': method}
    if extra_headers is not None:
        call_kwargs['extra_headers'] = extra_headers
    if method == 'GET':
        call_kwargs['params'] = json_body
        response = http_request(**call_kwargs)
    else:
        call_kwargs['json'] = json_body
        response = http_request(**call_kwargs)
    response = verify_rest_response(response, endpoint)
    js_dict = json.loads(response.text)
    parse_dict(js_dict=js_dict, message=response_proto)
    return response_proto