import base64
import warnings
from scribe import scribe
from thrift.transport import TTransport, TSocket
from thrift.protocol import TBinaryProtocol
from eventlet import GreenPile
def send_to_collector(self, span):
    self.pile.spawn(self._send, span)