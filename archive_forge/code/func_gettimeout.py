import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
def gettimeout(self):
    return self.socket.gettimeout()