from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def read_stdout(self, timeout=None):
    """Same as read_channel with channel=1."""
    return self.read_channel(STDOUT_CHANNEL, timeout=timeout)