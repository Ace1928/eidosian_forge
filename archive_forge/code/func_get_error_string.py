from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def get_error_string(self, e):
    """Get the string showing how 'e' would be formatted in test output.

        This is a little bit hacky, since it's designed to give consistent
        output regardless of Python version.

        In testtools, TestResult._exc_info_to_unicode is the point of dispatch
        between various different implementations of methods that format
        exceptions, so that's what we have to call. However, that method cares
        about stack traces and formats the exception class. We don't care
        about either of these, so we take its output and parse it a little.
        """
    error = TracebackContent((e.__class__, e, None), self).as_text()
    if error.startswith('Traceback (most recent call last):\n'):
        lines = error.splitlines(True)[1:]
        for i, line in enumerate(lines):
            if not line.startswith(' '):
                break
        error = ''.join(lines[i:])
    exc_class, error = error.split(': ', 1)
    return error