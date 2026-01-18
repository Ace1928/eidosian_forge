import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def discovery_required_for(self, operation_name):
    try:
        operation_model = self._service_model.operation_model(operation_name)
        return operation_model.endpoint_discovery.get('required', False)
    except OperationNotFoundError:
        return False