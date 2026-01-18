import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def test_files_skipped(self):
    tests = []
    result = StreamToDict(tests.append)
    result.startTestRun()
    result.status(file_name='some log.txt', file_bytes='', eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
    result.stopTestRun()
    self.assertThat(tests, HasLength(1))
    details = tests[0]['details']
    self.assertNotIn('some log.txt', details)