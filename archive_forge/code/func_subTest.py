from __future__ import unicode_literals
import contextlib
import difflib
import io
import os
import shutil
import subprocess
import sys
import unittest
import tempfile
@contextlib.contextmanager
def subTest(self, msg=None, **params):
    if sys.version_info < (3, 4, 0):
        yield None
    else:
        yield super(TestInvocations, self).subTest(msg=msg, **params)