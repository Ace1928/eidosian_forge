import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def create_request_object(request_dict):
    """
    This method takes a request dict and creates an AWSRequest object
    from it.

    :type request_dict: dict
    :param request_dict:  The request dict (created from the
        ``prepare_request_dict`` method).

    :rtype: ``botocore.awsrequest.AWSRequest``
    :return: An AWSRequest object based on the request_dict.

    """
    r = request_dict
    request_object = AWSRequest(method=r['method'], url=r['url'], data=r['body'], headers=r['headers'], auth_path=r.get('auth_path'))
    request_object.context = r['context']
    return request_object