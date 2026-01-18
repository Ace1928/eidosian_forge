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
def assertRaises(self, excClass, callableObj, *args, **kwargs):
    """Assert that a callable raises a particular exception.

        :param excClass: As for the except statement, this may be either an
            exception class, or a tuple of classes.
        :param callableObj: A callable, will be passed ``*args`` and
            ``**kwargs``.

        Returns the exception so that you can examine it.
        """
    try:
        callableObj(*args, **kwargs)
    except excClass as e:
        return e
    else:
        if getattr(excClass, '__name__', None) is not None:
            excName = excClass.__name__
        else:
            excName = str(excClass)
        raise self.failureException('%s not raised' % excName)