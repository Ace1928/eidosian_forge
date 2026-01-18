import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def exception_handler_v20(status_code, error_content):
    """Exception handler for API v2.0 client.

    This routine generates the appropriate Neutron exception according to
    the contents of the response body.

    :param status_code: HTTP error status code
    :param error_content: deserialized body of error response
    """
    error_dict = None
    request_ids = error_content.request_ids
    if isinstance(error_content, dict):
        error_dict = error_content.get('NeutronError')
    client_exc = None
    if error_dict:
        try:
            error_type = error_dict['type']
            error_message = error_dict['message']
            if error_dict['detail']:
                error_message += '\n' + error_dict['detail']
            client_exc = getattr(exceptions, '%sClient' % error_type, None)
        except Exception:
            error_message = '%s' % error_dict
    else:
        error_message = None
        if isinstance(error_content, dict):
            error_message = error_content.get('message')
        if not error_message:
            error_message = '%s-%s' % (status_code, error_content)
    if not client_exc:
        client_exc = exceptions.HTTP_EXCEPTION_MAP.get(status_code)
    if not client_exc:
        client_exc = exceptions.NeutronClientException
    raise client_exc(message=error_message, status_code=status_code, request_ids=request_ids)