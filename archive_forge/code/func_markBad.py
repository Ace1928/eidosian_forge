import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def markBad(self, mx):
    """
        Record that a mail exchange host is not currently functioning.

        @type mx: L{bytes}
        @param mx: The hostname of a mail exchange host.
        """
    self.badMXs[str(mx)] = self.clock.seconds() + self.timeOutBadMX