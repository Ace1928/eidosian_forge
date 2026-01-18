import unittest
from testtools import (
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.testresult.doubles import (
def getDetails(self):
    raise AttributeError('getDetails broke')