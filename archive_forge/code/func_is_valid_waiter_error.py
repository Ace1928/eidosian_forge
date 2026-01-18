import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
def is_valid_waiter_error(response):
    error = response.get('Error')
    if isinstance(error, dict) and 'Code' in error:
        return True
    return False