import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def do_brz(self, test_case, input, args):
    encoding = osutils.get_user_encoding()
    stdout = ui_testing.StringIOWithEncoding()
    stderr = ui_testing.StringIOWithEncoding()
    stdout.encoding = stderr.encoding = encoding
    handler = logging.StreamHandler(stderr)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger('')
    logger.addHandler(handler)
    try:
        retcode = test_case._run_bzr_core(args, encoding=encoding, stdin=input, stdout=stdout, stderr=stderr, working_dir=None)
    finally:
        logger.removeHandler(handler)
    return (retcode, stdout.getvalue(), stderr.getvalue())