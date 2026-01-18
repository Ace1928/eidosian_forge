import errno
from io import StringIO
from signal import SIGTERM
from types import TracebackType
from typing import Any, Iterable, List, Optional, TextIO, Tuple, Type, Union, cast
from attr import Factory, attrib, attrs
import twisted.trial.unittest
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.filepath import FilePath
from ...runner import _runner
from .._exit import ExitStatus
from .._pidfile import NonePIDFile, PIDFile
from .._runner import Runner
def test_killRequestedWithoutPIDFile(self) -> None:
    """
        L{Runner.killIfRequested} when C{kill} is true but C{pidFile} is
        L{nonePIDFile} exits with L{ExitStatus.EX_USAGE} and the expected
        message; and also doesn't indiscriminately murder anyone.
        """
    runner = Runner(reactor=MemoryReactor(), kill=True)
    runner.killIfRequested()
    self.assertEqual(self.kill.calls, [])
    self.assertEqual(self.exit.status, ExitStatus.EX_USAGE)
    self.assertEqual(self.exit.message, 'No PID file specified.')