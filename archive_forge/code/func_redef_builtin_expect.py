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
def redef_builtin_expect(self, cond):
    self.putln('#if %s' % cond)
    self.putln('    #undef likely')
    self.putln('    #undef unlikely')
    self.putln('    #define likely(x)   __builtin_expect(!!(x), 1)')
    self.putln('    #define unlikely(x) __builtin_expect(!!(x), 0)')
    self.putln('#endif')