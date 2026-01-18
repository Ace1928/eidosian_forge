from io import BytesIO, StringIO
from typing import IO, Any, List, Optional, Sequence, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._flatten import extractField
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._json import (
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
def savedJSONInvariants(testCase: TestCase, savedJSON: str) -> str:
    """
    Assert a few things about the result of L{eventAsJSON}, then return it.

    @param testCase: The L{TestCase} with which to perform the assertions.
    @param savedJSON: The result of L{eventAsJSON}.

    @return: C{savedJSON}

    @raise AssertionError: If any of the preconditions fail.
    """
    testCase.assertIsInstance(savedJSON, str)
    testCase.assertEqual(savedJSON.count('\n'), 0)
    return savedJSON