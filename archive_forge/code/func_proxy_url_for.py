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
def proxy_url_for(self, url):
    """Retrieves the corresponding proxy url for a given url."""
    parsed_url = urlparse(url)
    proxy = self._proxies.get(parsed_url.scheme)
    if proxy:
        proxy = self._fix_proxy_url(proxy)
    return proxy