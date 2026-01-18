from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def put_xdecref_clear(self, cname, type, clear_before_decref=False, nanny=True, have_gil=True):
    type.generate_xdecref_clear(self, cname, clear_before_decref=clear_before_decref, nanny=nanny, have_gil=have_gil)