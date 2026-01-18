import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
def test_exerciseHandleAccess_2(self):
    self.common_exerciseHandleAccess(cbInQue=True)