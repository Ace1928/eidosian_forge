import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_no_arg_named_message(self):
    """Ensure the __init__ and _fmt in errors do not have "message" arg.

        This test fails if __init__ or _fmt in errors has an argument
        named "message" as this can cause errors in some Python versions.
        Python 2.5 uses a slot for StandardError.message.
        See bug #603461
        """
    fmt_pattern = re.compile('%\\(message\\)[sir]')
    for c in errors.BzrError.__subclasses__():
        init = getattr(c, '__init__', None)
        fmt = getattr(c, '_fmt', None)
        if init:
            args = inspect.getfullargspec(init)[0]
            self.assertFalse('message' in args, 'Argument name "message" not allowed for "errors.%s.__init__"' % c.__name__)
        if fmt and fmt_pattern.search(fmt):
            self.assertFalse(True, '"message" not allowed in "errors.%s._fmt"' % c.__name__)