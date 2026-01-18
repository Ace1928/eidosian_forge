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
def split_suite_by_re(suite, pattern):
    """Split a test suite into two by a regular expression.

    :param suite: The suite to split.
    :param pattern: A regular expression string. Test ids that match this
        pattern will be in the first test suite returned, and the others in the
        second test suite returned.
    :return: A tuple of two test suites, where the first contains tests from
        suite matching pattern, and the second contains the remainder from
        suite. The order within each output suite is the same as it was in
        suite.
    """
    return split_suite_by_condition(suite, condition_id_re(pattern))