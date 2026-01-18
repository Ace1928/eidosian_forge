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
def test_add_expected_failure_details(self):
    """Test addExpectedFailure on a TestProtocolClient with details."""
    self.protocol.addExpectedFailure(self.test, details=self.sample_tb_details)
    self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('xfail: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_str_chunked + ']\n') % self.test.id())), Equals(_b(('xfail: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_repr_chunked + ']\n') % self.test.id()))))