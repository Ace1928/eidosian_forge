import errno
from functools import wraps
from os import getpid, name as SYSTEM_NAME
from typing import Any, Callable, Optional
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
import twisted.trial.unittest
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
from ...runner import _pidfile
from .._pidfile import (
def ifPlatformSupported(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for tests that are not expected to work on all platforms.

    Calling L{PIDFile.isRunning} currently raises L{NotImplementedError} on
    non-POSIX platforms.

    On an unsupported platform, we expect to see any test that calls
    L{PIDFile.isRunning} to raise either L{NotImplementedError}, L{SkipTest},
    or C{self.failureException}.
    (C{self.failureException} may occur in a test that checks for a specific
    exception but it gets NotImplementedError instead.)

    @param f: The test method to decorate.

    @return: The wrapped callable.
    """

    @wraps(f)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        supported = platform.getType() == 'posix'
        if supported:
            return f(self, *args, **kwargs)
        else:
            e = self.assertRaises((NotImplementedError, SkipTest, self.failureException), f, self, *args, **kwargs)
            if isinstance(e, NotImplementedError):
                self.assertTrue(str(e).startswith('isRunning is not implemented on '))
    return wrapper