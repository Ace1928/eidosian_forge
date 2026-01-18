import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def beginFileTransfer(self, file, consumer, transform=None):
    """
        Begin transferring a file

        @type file: Any file-like object
        @param file: The file object to read data from

        @type consumer: Any implementor of IConsumer
        @param consumer: The object to write data to

        @param transform: A callable taking one string argument and returning
        the same.  All bytes read from the file are passed through this before
        being written to the consumer.

        @rtype: C{Deferred}
        @return: A deferred whose callback will be invoked when the file has
        been completely written to the consumer. The last byte written to the
        consumer is passed to the callback.
        """
    self.file = file
    self.consumer = consumer
    self.transform = transform
    self.deferred = deferred = defer.Deferred()
    self.consumer.registerProducer(self, False)
    return deferred