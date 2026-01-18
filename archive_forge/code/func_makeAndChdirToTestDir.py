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
def makeAndChdirToTestDir(self):
    """See TestCaseWithMemoryTransport.makeAndChdirToTestDir().

        For TestCaseInTempDir we create a temporary directory based on the test
        name and then create two subdirs - test and home under it.
        """
    name_prefix = osutils.pathjoin(TestCaseWithMemoryTransport.TEST_ROOT, self._getTestDirPrefix())
    name = name_prefix
    for i in range(100):
        if os.path.exists(name):
            name = name_prefix + '_' + str(i)
        else:
            self.test_base_dir = name
            self.addCleanup(self.deleteTestDir)
            os.mkdir(self.test_base_dir)
            break
    self.permit_dir(self.test_base_dir)
    self.test_home_dir = self.test_base_dir + '/home'
    os.mkdir(self.test_home_dir)
    self.test_dir = self.test_base_dir + '/work'
    os.mkdir(self.test_dir)
    os.chdir(self.test_dir)
    with open(self.test_base_dir + '/name', 'w') as f:
        f.write(self.id())