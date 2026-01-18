import json
import logging
from mlflow.protos.databricks_pb2 import (
def get_http_status_code(self):
    return ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)