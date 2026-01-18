from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def read_channel(self, channel, timeout=0):
    """Read data from a channel."""
    if channel not in self._channels:
        ret = self.peek_channel(channel, timeout)
    else:
        ret = self._channels[channel]
    if channel in self._channels:
        del self._channels[channel]
    return ret