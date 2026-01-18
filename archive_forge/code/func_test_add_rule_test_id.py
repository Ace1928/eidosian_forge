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
def test_add_rule_test_id(self):
    nontest = LoggingStreamResult()
    test = LoggingStreamResult()
    router = StreamResultRouter(test)
    router.add_rule(nontest, 'test_id', test_id=None)
    router.status(test_id='foo', file_name='bar', file_bytes=b'')
    router.status(file_name='bar', file_bytes=b'')
    self.assertEqual([('status', 'foo', None, None, True, 'bar', b'', False, None, None, None)], test._events)
    self.assertEqual([('status', None, None, None, True, 'bar', b'', False, None, None, None)], nontest._events)