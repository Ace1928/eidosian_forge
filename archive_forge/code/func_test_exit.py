import io
import unittest
from testtools import PlaceHolder, TestCase
from testtools.compat import _b
from testtools.matchers import StartsWith
from testtools.testresult.doubles import StreamResult
import subunit
from subunit import run
from subunit.run import SubunitTestRunner
def test_exit(self):
    raise SystemExit(0)