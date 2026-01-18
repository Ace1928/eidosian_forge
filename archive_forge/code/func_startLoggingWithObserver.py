import errno
import sys
from io import BytesIO
from twisted.internet.testing import StringTransport
from twisted.protocols.amp import AMP
from twisted.trial._dist import (
from twisted.trial._dist.workertrial import WorkerLogObserver, main
from twisted.trial.unittest import TestCase
def startLoggingWithObserver(self, emit, setStdout):
    """
        Override C{startLoggingWithObserver} for not starting logging.
        """
    self.assertFalse(setStdout)