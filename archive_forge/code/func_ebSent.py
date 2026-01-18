import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def ebSent(err):
    """
            Called from data transport when there are errors during the
            transfer.
            """
    log.err(err, 'Unexpected error received during transfer:')
    if err.check(FTPCmdError):
        return err
    return (CNX_CLOSED_TXFR_ABORTED,)