import logging
import os
import os.path
import socket
import sys
import warnings
from base64 import b64encode
from urllib3 import PoolManager, Timeout, proxy_from_url
from urllib3.exceptions import (
from urllib3.exceptions import (
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import (
from urllib3.util.url import parse_url
import botocore.awsrequest
from botocore.compat import (
from botocore.exceptions import (
def proxy_headers_for(self, proxy_url):
    """Retrieves the corresponding proxy headers for a given proxy url."""
    headers = {}
    username, password = self._get_auth_from_url(proxy_url)
    if username and password:
        basic_auth = self._construct_basic_auth(username, password)
        headers['Proxy-Authorization'] = basic_auth
    return headers