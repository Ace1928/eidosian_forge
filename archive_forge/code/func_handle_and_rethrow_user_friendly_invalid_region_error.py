import time
import hashlib
from typing import List
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.utils.connection import get_response_object
def handle_and_rethrow_user_friendly_invalid_region_error(host, e):
    """
    Utility method which throws a more user-friendly error in case "name or
    service not known" error is received when sending a request.

    In most cases this error indicates user passed invalid ``region`` argument
    to the driver constructor.
    """
    msg = str(e).lower()
    error_messages_to_throw = ['name or service not known', 'nodename nor servname provided, or not known', 'getaddrinfo failed']
    if any([value for value in error_messages_to_throw if value in msg]):
        raise ValueError('Received "name or service not known" error when sending a request. This likely indicates invalid region argument was passed to the driver constructor.Used host: %s. Original error: %s' % (host, str(e)))
    raise e