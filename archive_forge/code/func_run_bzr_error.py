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
def run_bzr_error(self, error_regexes, *args, **kwargs):
    """Run brz, and check that stderr contains the supplied regexes

        :param error_regexes: Sequence of regular expressions which
            must each be found in the error output. The relative ordering
            is not enforced.
        :param args: command-line arguments for brz
        :param kwargs: Keyword arguments which are interpreted by run_brz
            This function changes the default value of retcode to be 3,
            since in most cases this is run when you expect brz to fail.

        :return: (out, err) The actual output of running the command (in case
            you want to do more inspection)

        Examples of use::

            # Make sure that commit is failing because there is nothing to do
            self.run_bzr_error(['no changes to commit'],
                               ['commit', '-m', 'my commit comment'])
            # Make sure --strict is handling an unknown file, rather than
            # giving us the 'nothing to do' error
            self.build_tree(['unknown'])
            self.run_bzr_error(
                ['Commit refused because there are unknown files'],
                ['commit', --strict', '-m', 'my commit comment'])
        """
    kwargs.setdefault('retcode', 3)
    kwargs['error_regexes'] = error_regexes
    out, err = self.run_bzr(*args, **kwargs)
    return (out, err)