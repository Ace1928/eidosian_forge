from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def expect_loop(self, searcher, timeout=-1, searchwindowsize=-1):
    """This is the common loop used inside expect. The 'searcher' should be
        an instance of searcher_re or searcher_string, which describes how and
        what to search for in the input.

        See expect() for other arguments, return value and exceptions. """
    exp = Expecter(self, searcher, searchwindowsize)
    return exp.expect_loop(timeout)