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
def suite_matches_id_list(test_suite, id_list):
    """Warns about tests not appearing or appearing more than once.

    :param test_suite: A TestSuite object.
    :param test_id_list: The list of test ids that should be found in
         test_suite.

    :return: (absents, duplicates) absents is a list containing the test found
        in id_list but not in test_suite, duplicates is a list containing the
        tests found multiple times in test_suite.

    When using a prefined test id list, it may occurs that some tests do not
    exist anymore or that some tests use the same id. This function warns the
    tester about potential problems in his workflow (test lists are volatile)
    or in the test suite itself (using the same id for several tests does not
    help to localize defects).
    """
    tests = dict()
    for test in iter_suite_tests(test_suite):
        id = test.id()
        tests[id] = tests.get(id, 0) + 1
    not_found = []
    duplicates = []
    for id in id_list:
        occurs = tests.get(id, 0)
        if not occurs:
            not_found.append(id)
        elif occurs > 1:
            duplicates.append(id)
    return (not_found, duplicates)