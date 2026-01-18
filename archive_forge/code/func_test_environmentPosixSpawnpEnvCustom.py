import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
@onlyOnPOSIX
def test_environmentPosixSpawnpEnvCustom(self):
    """
        The user-specified environment without extra variables from parent process is passed to the
        spawned process, when C{env} is set to a dictionary.

        In this case posix_spawnp is used as the backend for spawning processes.
        """
    return self.checkSpawnProcessEnvironmentWithPosixSpawnp({'env': {'MYENV1': 'myvalue1'}}, {'MYENV1': 'myvalue1'})