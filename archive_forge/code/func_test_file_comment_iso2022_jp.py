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
def test_file_comment_iso2022_jp(self):
    """Control character escapes must be preserved if valid encoding"""
    example_text, _ = self._get_sample_text('iso2022_jp')
    textoutput = self._test_external_case(coding='iso2022_jp', testline="self.fail('Simple') # %s" % example_text)
    self.assertIn(self._as_output(example_text), textoutput)