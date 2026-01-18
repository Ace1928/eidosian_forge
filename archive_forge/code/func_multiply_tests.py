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
def multiply_tests(tests, scenarios, result):
    """Multiply tests_list by scenarios into result.

    This is the core workhorse for test parameterisation.

    Typically the load_tests() method for a per-implementation test suite will
    call multiply_tests and return the result.

    :param tests: The tests to parameterise.
    :param scenarios: The scenarios to apply: pairs of (scenario_name,
        scenario_param_dict).
    :param result: A TestSuite to add created tests to.

    This returns the passed in result TestSuite with the cross product of all
    the tests repeated once for each scenario.  Each test is adapted by adding
    the scenario name at the end of its id(), and updating the test object's
    __dict__ with the scenario_param_dict.

    >>> import breezy.tests.test_sampler
    >>> r = multiply_tests(
    ...     breezy.tests.test_sampler.DemoTest('test_nothing'),
    ...     [('one', dict(param=1)),
    ...      ('two', dict(param=2))],
    ...     TestUtil.TestSuite())
    >>> tests = list(iter_suite_tests(r))
    >>> len(tests)
    2
    >>> tests[0].id()
    'breezy.tests.test_sampler.DemoTest.test_nothing(one)'
    >>> tests[0].param
    1
    >>> tests[1].param
    2
    """
    for test in iter_suite_tests(tests):
        apply_scenarios(test, scenarios, result)
    return result