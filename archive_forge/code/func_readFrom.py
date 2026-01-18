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
def readFrom(self, pipe):
    """Blocking convenience API to parse an entire stream.

        :param pipe: A file-like object supporting __iter__.
        :return: None.
        """
    for line in pipe:
        self.lineReceived(line)
    self.lostConnection()