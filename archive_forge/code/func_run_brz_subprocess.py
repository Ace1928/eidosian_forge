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
def run_brz_subprocess(self, *args, **kwargs):
    """Run brz in a subprocess for testing.

        This starts a new Python interpreter and runs brz in there.
        This should only be used for tests that have a justifiable need for
        this isolation: e.g. they are testing startup time, or signal
        handling, or early startup code, etc.  Subprocess code can't be
        profiled or debugged so easily.

        :keyword retcode: The status code that is expected.  Defaults to 0.  If
            None is supplied, the status code is not checked.
        :keyword env_changes: A dictionary which lists changes to environment
            variables. A value of None will unset the env variable.
            The values must be strings. The change will only occur in the
            child, so you don't need to fix the environment after running.
        :keyword universal_newlines: Convert CRLF => LF
        :keyword allow_plugins: By default the subprocess is run with
            --no-plugins to ensure test reproducibility. Also, it is possible
            for system-wide plugins to create unexpected output on stderr,
            which can cause unnecessary test failures.
        """
    env_changes = kwargs.get('env_changes', None)
    working_dir = kwargs.get('working_dir', None)
    allow_plugins = kwargs.get('allow_plugins', False)
    if len(args) == 1:
        if isinstance(args[0], list):
            args = args[0]
        elif isinstance(args[0], str):
            args = list(shlex.split(args[0]))
    else:
        raise ValueError('passing varargs to run_brz_subprocess')
    process = self.start_brz_subprocess(args, env_changes=env_changes, working_dir=working_dir, allow_plugins=allow_plugins)
    supplied_retcode = kwargs.get('retcode', 0)
    return self.finish_brz_subprocess(process, retcode=supplied_retcode, universal_newlines=kwargs.get('universal_newlines', False), process_args=args)