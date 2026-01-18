import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def switch_to_virtual_host_style(request, signature_version, default_endpoint_url=None, **kwargs):
    """
    This is a handler to force virtual host style s3 addressing no matter
    the signature version (which is taken in consideration for the default
    case). If the bucket is not DNS compatible an InvalidDNSName is thrown.

    :param request: A AWSRequest object that is about to be sent.
    :param signature_version: The signature version to sign with
    :param default_endpoint_url: The endpoint to use when switching to a
        virtual style. If None is supplied, the virtual host will be
        constructed from the url of the request.
    """
    if request.auth_path is not None:
        return
    elif _is_get_bucket_location_request(request):
        logger.debug('Request is GetBucketLocation operation, not checking for DNS compatibility.')
        return
    parts = urlsplit(request.url)
    request.auth_path = parts.path
    path_parts = parts.path.split('/')
    if default_endpoint_url is None:
        default_endpoint_url = parts.netloc
    if len(path_parts) > 1:
        bucket_name = path_parts[1]
        if not bucket_name:
            return
        logger.debug('Checking for DNS compatible bucket for: %s', request.url)
        if check_dns_name(bucket_name):
            if len(path_parts) == 2:
                if request.auth_path[-1] != '/':
                    request.auth_path += '/'
            path_parts.remove(bucket_name)
            path = '/'.join(path_parts) or '/'
            global_endpoint = default_endpoint_url
            host = bucket_name + '.' + global_endpoint
            new_tuple = (parts.scheme, host, path, parts.query, '')
            new_uri = urlunsplit(new_tuple)
            request.url = new_uri
            logger.debug('URI updated to: %s', new_uri)
        else:
            raise InvalidDNSNameError(bucket_name=bucket_name)