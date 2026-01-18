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
def getRelayed(self):
    """
        Return the base filenames of messages in the process of being relayed.

        @rtype: L{list} of L{bytes}
        @return: The base filenames of messages in the process of being
            relayed.
        """
    return self.relayed.keys()