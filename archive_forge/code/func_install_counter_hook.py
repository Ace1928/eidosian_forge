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
def install_counter_hook(self, hooks, name, counter_name=None):
    """Install a counting hook.

        Any hook can be counted as long as it doesn't need to return a value.

        :param hooks: Where the hook should be installed.

        :param name: The hook name that will be counted.

        :param counter_name: The counter identifier in ``_counters``, defaults
            to ``name``.
        """
    _counters = self._counters
    if counter_name is None:
        counter_name = name
    if counter_name in _counters:
        raise AssertionError('%s is already used as a counter name' % (counter_name,))
    _counters[counter_name] = 0
    self.addDetail(counter_name, content.Content(content.UTF8_TEXT, lambda: [b'%d' % (_counters[counter_name],)]))

    def increment_counter(*args, **kwargs):
        _counters[counter_name] += 1
    label = 'count {} calls'.format(counter_name)
    hooks.install_named_hook(name, increment_counter, label)
    self.addCleanup(hooks.uninstall_named_hook, name, label)