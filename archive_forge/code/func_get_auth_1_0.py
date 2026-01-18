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
def get_auth_1_0(url, user, key, snet, **kwargs):
    cacert = kwargs.get('cacert', None)
    insecure = kwargs.get('insecure', False)
    cert = kwargs.get('cert')
    cert_key = kwargs.get('cert_key')
    timeout = kwargs.get('timeout', None)
    parsed, conn = http_connection(url, cacert=cacert, insecure=insecure, cert=cert, cert_key=cert_key, timeout=timeout)
    method = 'GET'
    headers = {'X-Auth-User': user, 'X-Auth-Key': key}
    conn.request(method, parsed.path, '', headers)
    resp = conn.getresponse()
    body = resp.read()
    resp.close()
    conn.close()
    http_log((url, method), headers, resp, body)
    url = resp.getheader('x-storage-url')
    if resp.status < 200 or resp.status >= 300 or (body and (not url)):
        raise ClientException.from_response(resp, 'Auth GET failed', body)
    if snet:
        parsed = list(urlparse(url))
        netloc = parsed[1]
        parsed[1] = 'snet-' + netloc
        url = urlunparse(parsed)
    token = resp.getheader('x-storage-token', resp.getheader('x-auth-token'))
    return (url, token)