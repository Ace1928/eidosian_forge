import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def createProcesses(self, usePTY=0):
    scriptPath = b'twisted.test.process_reader'
    for num in (0, 1):
        self.pp[num] = TwoProcessProtocol()
        self.pp[num].num = num
        p = reactor.spawnProcess(self.pp[num], pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, usePTY=usePTY)
        self.processes[num] = p