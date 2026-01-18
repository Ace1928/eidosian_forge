from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def readline_channel(self, channel, timeout=None):
    """Read a line from a channel."""
    if timeout is None:
        timeout = float('inf')
    start = time.time()
    while self.is_open() and time.time() - start < timeout:
        if channel in self._channels:
            data = self._channels[channel]
            if '\n' in data:
                index = data.find('\n')
                ret = data[:index]
                data = data[index + 1:]
                if data:
                    self._channels[channel] = data
                else:
                    del self._channels[channel]
                return ret
        self.update(timeout=timeout - time.time() + start)