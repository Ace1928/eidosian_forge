from sys import stderr, stdout
from typing import Callable, Dict, List, Optional, TextIO, Tuple
import twisted.trial.unittest
from twisted.copyright import version
from twisted.internet import reactor
from twisted.internet.interfaces import IReactorCore
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.usage import UsageError
from ...reactors import NoSuchReactor
from ...runner._exit import ExitStatus
from ...runner.test.test_runner import DummyExit
from ...service import ServiceMaker
from ...twist import _options
from .._options import TwistOptions
def test_logFileCantOpen(self) -> None:
    """
        L{TwistOptions.opt_log_file} exits with L{ExitStatus.EX_IOERR} if
        unable to open the log file due to an L{EnvironmentError}.
        """
    self.patchExit()
    self.patchOpen()
    options = TwistOptions()
    options.opt_log_file('nocanopen')
    self.assertEquals(self.exit.status, ExitStatus.EX_IOERR)
    self.assertIsNotNone(self.exit.message)
    self.assertTrue(self.exit.message.startswith("Unable to open log file 'nocanopen': "))