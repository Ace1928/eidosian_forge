import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
def post_object(self, container, obj, headers, response_dict=None):
    """Wrapper for :func:`post_object`"""
    return self._retry(None, post_object, container, obj, headers, response_dict=response_dict)