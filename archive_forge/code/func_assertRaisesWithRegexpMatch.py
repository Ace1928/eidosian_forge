import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def assertRaisesWithRegexpMatch(self, exception, regexp, function, *params, **kwargs):
    """Check that exception is raised and text matches regular expression.

        Args:
          exception: Exception type that is expected.
          regexp: String regular expression that is expected in error message.
          function: Callable to test.
          params: Parameters to forward to function.
          kwargs: Keyword arguments to forward to function.
        """
    try:
        function(*params, **kwargs)
        self.fail('Expected exception %s was not raised' % exception.__name__)
    except exception as err:
        match = bool(re.match(regexp, str(err)))
        self.assertTrue(match, 'Expected match "%s", found "%s"' % (regexp, err))