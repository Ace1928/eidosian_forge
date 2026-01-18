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
def test_all_terminal_states_reported(self):
    tests = []
    result = StreamToDict(tests.append)
    result.startTestRun()
    result.status('success', 'success')
    result.status('skip', 'skip')
    result.status('exists', 'exists')
    result.status('fail', 'fail')
    result.status('xfail', 'xfail')
    result.status('uxsuccess', 'uxsuccess')
    self.assertThat(tests, HasLength(6))
    self.assertEqual(['success', 'skip', 'exists', 'fail', 'xfail', 'uxsuccess'], [test['id'] for test in tests])
    result.stopTestRun()
    self.assertThat(tests, HasLength(6))