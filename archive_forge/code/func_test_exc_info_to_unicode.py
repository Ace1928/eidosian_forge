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
def test_exc_info_to_unicode(self):
    test = make_erroring_test()
    exc_info = make_exception_info(RuntimeError, 'foo')
    result = self.makeResult()
    text_traceback = result._exc_info_to_unicode(exc_info, test)
    self.assertEqual(TracebackContent(exc_info, test).as_text(), text_traceback)