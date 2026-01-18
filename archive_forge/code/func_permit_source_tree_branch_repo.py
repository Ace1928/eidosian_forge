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
def permit_source_tree_branch_repo(self):
    """Permit the source tree brz is running from to be opened.

        Some code such as breezy.version attempts to read from the brz branch
        that brz is executing from (if any). This method permits that directory
        to be used in the test suite.
        """
    path = self.get_source_path()
    self.record_directory_isolation()
    try:
        try:
            workingtree.WorkingTree.open(path)
        except (errors.NotBranchError, errors.NoWorkingTree):
            raise TestSkipped('Needs a working tree of brz sources')
    finally:
        self.enable_directory_isolation()