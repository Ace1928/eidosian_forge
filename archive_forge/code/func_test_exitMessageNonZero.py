from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit
def test_exitMessageNonZero(self) -> None:
    """
        L{exit} given a non-zero status code writes the given message to
        standard error.
        """
    out = StringIO()
    self.patch(_exit, 'stderr', out)
    message = 'Hello, world.'
    exit(64, message)
    self.assertEqual(out.getvalue(), message + '\n')