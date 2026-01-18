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
def check_fail_or_uxsuccess(self, as_fail, error_message=None):
    details = {}
    if error_message is not None:
        details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b(error_message)])
    if isinstance(self.client, ExtendedTestResult):
        value = details
    else:
        value = None
    if as_fail:
        self.client._events[1] = self.client._events[1][:2]
        self.assertEqual([('startTest', self.test), ('addFailure', self.test), ('stopTest', self.test)], self.client._events)
    elif value:
        self.assertEqual([('startTest', self.test), ('addUnexpectedSuccess', self.test, value), ('stopTest', self.test)], self.client._events)
    else:
        self.assertEqual([('startTest', self.test), ('addUnexpectedSuccess', self.test), ('stopTest', self.test)], self.client._events)