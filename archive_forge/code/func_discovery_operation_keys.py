import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
@CachedProperty
def discovery_operation_keys(self):
    discovery_operation = self._service_model.endpoint_discovery_operation
    keys = []
    if discovery_operation.input_shape:
        keys = list(discovery_operation.input_shape.members.keys())
    return keys