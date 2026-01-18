import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testConvertIdThatNeedsEscaping(self):
    self._DoTestConvertIdToHeader('~tilde1', '<%s+%%7Etilde1>')