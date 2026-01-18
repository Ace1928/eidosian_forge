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
def assertMultiLineEqual(self, first, second, msg=None):
    """Assert that two multi-line strings are equal."""
    assert isinstance(first, types.StringTypes), 'First argument is not a string: %r' % (first,)
    assert isinstance(second, types.StringTypes), 'Second argument is not a string: %r' % (second,)
    if first == second:
        return
    if msg:
        raise self.failureException(msg)
    failure_message = ['\n']
    for line in difflib.ndiff(first.splitlines(True), second.splitlines(True)):
        failure_message.append(line)
        if not line.endswith('\n'):
            failure_message.append('\n')
    raise self.failureException(''.join(failure_message))