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
def test_installCorrectReactor(self) -> None:
    """
        L{TwistOptions.installReactor} installs the chosen reactor after the
        command line options have been parsed.
        """
    self.patchInstallReactor()
    options = TwistOptions()
    options.subCommand = 'test-subcommand'
    options.parseOptions(['--reactor=fusion'])
    self.assertEqual(set(self.installedReactors), {'fusion'})