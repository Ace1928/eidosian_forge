import os
import pickle
from twisted.internet.address import UNIXAddress
from twisted.mail import smtp
from twisted.python import log
def startMessage(self, user):
    """
        Create an envelope and a message receiver for the relay queue.

        @type user: L{User}
        @param user: A user.

        @rtype: L{IMessage <smtp.IMessage>}
        @return: A message receiver.
        """
    queue = self.service.queue
    envelopeFile, smtpMessage = queue.createNewMessage()
    with envelopeFile:
        log.msg(f'Queueing mail {str(user.orig)!r} -> {str(user.dest)!r}')
        pickle.dump([str(user.orig), str(user.dest)], envelopeFile)
    return smtpMessage