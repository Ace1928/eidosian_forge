import errno
import sys
from io import BytesIO
from twisted.internet.testing import StringTransport
from twisted.protocols.amp import AMP
from twisted.trial._dist import (
from twisted.trial._dist.workertrial import WorkerLogObserver, main
from twisted.trial.unittest import TestCase
def test_forwardCommand(self):
    """
        L{main} forwards data from its input stream to a L{WorkerProtocol}
        instance which writes data to the output stream.
        """
    client = FakeAMP()
    clientTransport = StringTransport()
    client.makeConnection(clientTransport)
    client.callRemote(workercommands.Run, testCase='doesntexist')
    self.readStream = clientTransport.io
    self.readStream.seek(0, 0)
    main(self.fdopen)
    self.assertIn(b'StreamOpen', self.writeStream.getvalue())