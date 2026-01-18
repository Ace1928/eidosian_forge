import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
def sort_tests(self):
    self._tests = sorted_tests(self, True)