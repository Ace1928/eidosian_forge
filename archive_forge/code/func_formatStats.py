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
def formatStats(self):
    self._stream.write('Total tests:   %5d\n' % self.total_tests)
    self._stream.write('Passed tests:  %5d\n' % self.passed_tests)
    self._stream.write('Failed tests:  %5d\n' % self.failed_tests)
    self._stream.write('Skipped tests: %5d\n' % self.skipped_tests)
    tags = sorted(self.seen_tags)
    self._stream.write('Seen tags: %s\n' % ', '.join(tags))