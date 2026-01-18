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
def get_py_string_const(self, text, identifier=None, is_str=False, unicode_value=None):
    return self.globalstate.get_py_string_const(text, identifier, is_str, unicode_value).cname