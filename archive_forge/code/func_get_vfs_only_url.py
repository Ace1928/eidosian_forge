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
def get_vfs_only_url(self, relpath=None):
    """Get a URL (or maybe a path for the plain old vfs transport.

        This will never be a smart protocol.  It always has all the
        capabilities of the local filesystem, but it might actually be a
        MemoryTransport or some other similar virtual filesystem.

        This is the backing transport (if any) of the server returned by
        get_url and get_readonly_url.

        :param relpath: provides for clients to get a path relative to the base
            url.  These should only be downwards relative, not upwards.
        :return: A URL
        """
    base = self.get_vfs_only_server().get_url()
    return self._adjust_url(base, relpath)