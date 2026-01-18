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
def http_connection(self, url=None):
    return http_connection(url if url else self.url, cacert=self.cacert, insecure=self.insecure, cert=self.cert, cert_key=self.cert_key, ssl_compression=self.ssl_compression, timeout=self.timeout)