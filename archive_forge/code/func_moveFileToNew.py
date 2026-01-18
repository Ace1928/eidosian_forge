import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
def moveFileToNew(self):
    """
        Place the message in the I{new/} directory, add it to the mailbox and
        fire the deferred to indicate that the task has completed
        successfully.
        """
    while True:
        newname = os.path.join(self.mbox.path, 'new', _generateMaildirName())
        try:
            self.osrename(self.tmpname, newname)
            break
        except OSError as e:
            err, estr = e.args
            import errno
            if err != errno.EEXIST:
                self.fail()
                newname = None
                break
    if newname is not None:
        self.mbox.list.append(newname)
        self.defer.callback(None)
        self.defer = None