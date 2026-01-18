import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
def test_clientRekey(self):
    """
        After a client-initiated rekey is completed, application data continues
        to be passed over the SSH connection.
        """
    process = ConchTestOpenSSHProcess()
    d = self.execute('', process, '-o RekeyLimit=2K')

    def finished(result):
        expectedResult = '\n'.join(['line #%02d' % (i,) for i in range(60)]) + '\n'
        expectedResult = expectedResult.encode('utf-8')
        self.assertEqual(result, expectedResult)
    d.addCallback(finished)
    return d