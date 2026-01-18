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
def missing_test(plan_start):
    output.status(test_id='test %d' % plan_start, test_status='fail', runnable=False, mime_type=UTF8_TEXT, eof=True, file_name='tap meta', file_bytes=b'test missing from TAP output')