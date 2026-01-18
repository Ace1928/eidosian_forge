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
def safe_value(name, value):
    """
    Only show up to logger_settings['reveal_sensitive_prefix'] characters
    from a sensitive header.

    :param name: Header name
    :param value: Header value
    :return: Safe header value
    """
    if name.lower() in LOGGER_SENSITIVE_HEADERS:
        prefix_length = logger_settings.get('reveal_sensitive_prefix', 16)
        prefix_length = int(min(prefix_length, len(value) ** 2 / 32, len(value) / 2))
        redacted_value = value[0:prefix_length]
        return redacted_value + '...'
    return value