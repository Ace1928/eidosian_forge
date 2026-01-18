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
def reinvoke_for_tests(suite):
    """Take suite and start up one runner per CPU using subprocess().

    :return: An iterable of TestCase-like objects which can each have
        run(result) called on them to feed tests to result.
    """
    concurrency = osutils.local_concurrency()
    result = []
    from subunit import ProtocolTestCase

    class TestInSubprocess(ProtocolTestCase):

        def __init__(self, process, name):
            ProtocolTestCase.__init__(self, process.stdout)
            self.process = process
            self.process.stdin.close()
            self.name = name

        def run(self, result):
            try:
                ProtocolTestCase.run(self, result)
            finally:
                self.process.wait()
                os.unlink(self.name)
    test_blocks = partition_tests(suite, concurrency)
    for process_tests in test_blocks:
        bzr_path = os.path.dirname(os.path.dirname(breezy.__file__)) + '/bzr'
        if not os.path.isfile(bzr_path):
            bzr_path = sys.argv[0]
        bzr_path = [bzr_path]
        fd, test_list_file_name = tempfile.mkstemp()
        test_list_file = os.fdopen(fd, 'wb', 1)
        for test in process_tests:
            test_list_file.write(test.id() + '\n')
        test_list_file.close()
        try:
            argv = bzr_path + ['selftest', '--load-list', test_list_file_name, '--subunit']
            if '--no-plugins' in sys.argv:
                argv.append('--no-plugins')
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
            test = TestInSubprocess(process, test_list_file_name)
            result.append(test)
        except:
            os.unlink(test_list_file_name)
            raise
    return result