import datetime
import io
import os
import tempfile
import unittest
from io import BytesIO
from testtools import PlaceHolder, TestCase, TestResult, skipIf
from testtools.compat import _b, _u
from testtools.content import Content, TracebackContent, text_content
from testtools.content_type import ContentType
from testtools.matchers import Contains, Equals, MatchesAny
import iso8601
import subunit
from subunit.tests import (_remote_exception_repr,
def test_join_dir(self):
    sibling = subunit.join_dir(__file__, 'foo')
    filedir = os.path.abspath(os.path.dirname(__file__))
    expected = os.path.join(filedir, 'foo')
    self.assertEqual(sibling, expected)