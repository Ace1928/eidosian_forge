import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def assertRaisesWithLiteralMatch(self, expected_exception, expected_exception_message, callable_obj, *args, **kwargs):
    """Asserts that the message in a raised exception equals the given string.

    Unlike assertRaisesWithRegexpMatch this method takes a literal string, not
    a regular expression.

    Args:
      expected_exception: Exception class expected to be raised.
      expected_exception_message: String message expected in the raised
        exception.  For a raise exception e, expected_exception_message must
        equal str(e).
      callable_obj: Function to be called.
      args: Extra args.
      kwargs: Extra kwargs.
    """
    try:
        callable_obj(*args, **kwargs)
    except expected_exception as err:
        actual_exception_message = str(err)
        self.assert_(expected_exception_message == actual_exception_message, 'Exception message does not match.\nExpected: %r\nActual: %r' % (expected_exception_message, actual_exception_message))
    else:
        self.fail(expected_exception.__name__ + ' not raised')