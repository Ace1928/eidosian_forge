import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def set_http_proxy(self, proxy_url):
    """
        Set a HTTP / HTTPS proxy which will be used with this connection.

        :param proxy_url: Proxy URL (e.g. http://<hostname>:<port> without
                          authentication and
                          <scheme>://<username>:<password>@<hostname>:<port>
                          for basic auth authentication information.
        :type proxy_url: ``str``
        """
    self.proxy_url = proxy_url
    self.connection.set_http_proxy(proxy_url=proxy_url)