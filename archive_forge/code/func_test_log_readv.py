from breezy import transport
from breezy.tests import TestCaseWithMemoryTransport
from breezy.trace import mutter
from breezy.transport.log import TransportLogDecorator
def test_log_readv(self):
    base_transport = DummyReadvTransport()
    logging_transport = TransportLogDecorator('log+dummy:///', _decorated=base_transport)
    result = base_transport.readv('foo', [(0, 10)])
    next(result)
    result = logging_transport.readv('foo', [(0, 10)])
    self.assertEqual(list(result), [(0, 'abcdefghij')])