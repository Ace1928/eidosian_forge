import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def finish_brz_subprocess(self, process, retcode=0, send_signal=None, universal_newlines=False, process_args=None):
    """Finish the execution of process.

        :param process: the Popen object returned from start_brz_subprocess.
        :param retcode: The status code that is expected.  Defaults to 0.  If
            None is supplied, the status code is not checked.
        :param send_signal: an optional signal to send to the process.
        :param universal_newlines: Convert CRLF => LF
        :returns: (stdout, stderr)
        """
    if send_signal is not None:
        os.kill(process.pid, send_signal)
    out, err = process.communicate()
    if universal_newlines:
        out = out.replace(b'\r\n', b'\n')
        err = err.replace(b'\r\n', b'\n')
    if retcode is not None and retcode != process.returncode:
        if process_args is None:
            process_args = '(unknown args)'
        trace.mutter('Output of brz %r:\n%s', process_args, out)
        trace.mutter('Error for brz %r:\n%s', process_args, err)
        self.fail('Command brz %r failed with retcode %d != %d' % (process_args, retcode, process.returncode))
    return [out, err]