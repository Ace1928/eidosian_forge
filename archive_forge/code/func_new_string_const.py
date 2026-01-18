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
def new_string_const(self, text, byte_string):
    cname = self.new_string_const_cname(byte_string)
    c = StringConst(cname, text, byte_string)
    self.string_const_index[byte_string] = c
    return c