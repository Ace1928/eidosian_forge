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
def check_success_or_xfail(self, as_success, error_message=None):
    if as_success:
        self.assertEqual([('startTest', self.test), ('addSuccess', self.test), ('stopTest', self.test)], self.client._events)
    else:
        details = {}
        if error_message is not None:
            details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b(error_message)])
        if isinstance(self.client, ExtendedTestResult):
            value = details
        elif error_message is not None:
            value = subunit.RemoteError(details_to_str(details))
        else:
            value = subunit.RemoteError()
        self.assertEqual([('startTest', self.test), ('addExpectedFailure', self.test, value), ('stopTest', self.test)], self.client._events)