from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def get_websocket_url(url):
    parsed_url = urlparse(url)
    parts = list(parsed_url)
    if parsed_url.scheme == 'http':
        parts[0] = 'ws'
    elif parsed_url.scheme == 'https':
        parts[0] = 'wss'
    return urlunparse(parts)