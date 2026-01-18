from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatAttributeSubscript(self) -> None:
    """
        L{formatEvent} will format subscripts of attributes per PEP 3101.
        """

    class Example(object):
        config: Dict[str, str] = dict(foo='bar', baz='qux')
    self.assertEqual('bar qux', self.format('{example.config[foo]} {example.config[baz]}', example=Example()))