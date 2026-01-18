import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def assert_response(response_dict, status_code):
    """
    Basic checks that the response is of the type
    the client is expecting
    """
    if response_dict['type'] == LXD_ERROR_STATUS_RESP:
        raise LXDAPIException(message='response type is error', response_dict=response_dict)
    if response_dict['status_code'] != status_code:
        msg = 'Status code should be {}         but is {}'.format(status_code, response_dict['status_code'])
        raise LXDAPIException(message=msg, response_dict=response_dict)