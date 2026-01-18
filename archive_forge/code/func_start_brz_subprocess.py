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
def start_brz_subprocess(self, process_args, env_changes=None, skip_if_plan_to_signal=False, working_dir=None, allow_plugins=False, stderr=subprocess.PIPE):
    """Start brz in a subprocess for testing.

        This starts a new Python interpreter and runs brz in there.
        This should only be used for tests that have a justifiable need for
        this isolation: e.g. they are testing startup time, or signal
        handling, or early startup code, etc.  Subprocess code can't be
        profiled or debugged so easily.

        :param process_args: a list of arguments to pass to the brz executable,
            for example ``['--version']``.
        :param env_changes: A dictionary which lists changes to environment
            variables. A value of None will unset the env variable.
            The values must be strings. The change will only occur in the
            child, so you don't need to fix the environment after running.
        :param skip_if_plan_to_signal: raise TestSkipped when true and system
            doesn't support signalling subprocesses.
        :param allow_plugins: If False (default) pass --no-plugins to brz.
        :param stderr: file to use for the subprocess's stderr.  Valid values
            are those valid for the stderr argument of `subprocess.Popen`.
            Default value is ``subprocess.PIPE``.

        :returns: Popen object for the started process.
        """
    if skip_if_plan_to_signal:
        if os.name != 'posix':
            raise TestSkipped('Sending signals not supported')
    if env_changes is None:
        env_changes = {}
    if site.USER_BASE is not None:
        env_changes['PYTHONUSERBASE'] = site.USER_BASE
    if 'PYTHONPATH' not in env_changes:
        env_changes['PYTHONPATH'] = ':'.join(sys.path)
    old_env = {}

    def cleanup_environment():
        for env_var, value in env_changes.items():
            old_env[env_var] = osutils.set_or_unset_env(env_var, value)

    def restore_environment():
        for env_var, value in old_env.items():
            osutils.set_or_unset_env(env_var, value)
    cwd = None
    if working_dir is not None:
        cwd = osutils.getcwd()
        os.chdir(working_dir)
    try:
        cleanup_environment()
        self._add_subprocess_log(trace._get_brz_log_filename())
        command = self.get_brz_command()
        if not allow_plugins:
            command.append('--no-plugins')
        command.extend(process_args)
        process = self._popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
    finally:
        restore_environment()
        if cwd is not None:
            os.chdir(cwd)
    return process