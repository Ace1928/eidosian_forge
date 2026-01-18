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
def skip_quoted_bracket(self, keyword):
    self.protocol.lineReceived(_b('%s mcdonalds farm [\n' % keyword))
    self.protocol.lineReceived(_b(' ]\n'))
    self.protocol.lineReceived(_b(']\n'))
    self.assertSkip(_b(']\n'))