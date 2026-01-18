import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def endDetails(self):
    """The end of a details section has been reached."""
    self.parser._state = self.parser._outside_test
    self.parser.current_test_description = None
    self._report_outcome()
    self.parser.client.stopTest(self.parser._current_test)