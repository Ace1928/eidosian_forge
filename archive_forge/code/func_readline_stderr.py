from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def readline_stderr(self, timeout=None):
    """Same as readline_channel with channel=2."""
    return self.readline_channel(STDERR_CHANNEL, timeout=timeout)