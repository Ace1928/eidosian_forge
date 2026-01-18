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
def conditionally_calculate_md5(params, **kwargs):
    """Only add a Content-MD5 if the system supports it."""
    body = params['body']
    checksum_context = params.get('context', {}).get('checksum', {})
    checksum_algorithm = checksum_context.get('request_algorithm')
    if checksum_algorithm and checksum_algorithm != 'conditional-md5':
        return
    if _has_checksum_header(params):
        return
    if _is_s3express_request(params):
        return
    if MD5_AVAILABLE and body is not None:
        md5_digest = calculate_md5(body, **kwargs)
        params['headers']['Content-MD5'] = md5_digest