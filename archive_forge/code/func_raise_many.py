from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def raise_many(ignored):
    try:
        1 / 0
    except Exception:
        exc_info1 = sys.exc_info()
    try:
        1 / 0
    except Exception:
        exc_info2 = sys.exc_info()
    raise MultipleExceptions(exc_info1, exc_info2)