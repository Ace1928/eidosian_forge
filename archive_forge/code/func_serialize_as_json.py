import json
import logging
from mlflow.protos.databricks_pb2 import (
def serialize_as_json(self):
    exception_dict = {'error_code': self.error_code, 'message': self.message}
    exception_dict.update(self.json_kwargs)
    return json.dumps(exception_dict)