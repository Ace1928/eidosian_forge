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
def scrub_headers(headers):
    """
    Redact header values that can contain sensitive information that
    should not be logged.

    :param headers: Either a dict or an iterable of two-element tuples
    :return: Safe dictionary of headers with sensitive information removed
    """
    if isinstance(headers, dict):
        headers = headers.items()
    headers = [(parse_header_string(key), parse_header_string(val)) for key, val in headers]
    if not logger_settings.get('redact_sensitive_headers', True):
        return dict(headers)
    if logger_settings.get('reveal_sensitive_prefix', 16) < 0:
        logger_settings['reveal_sensitive_prefix'] = 16
    return {key: safe_value(key, val) for key, val in headers}