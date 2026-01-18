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
def put_cached_builtin_init(self, pos, name, cname):
    w = self.parts['cached_builtins']
    interned_cname = self.get_interned_identifier(name).cname
    self.use_utility_code(UtilityCode.load_cached('GetBuiltinName', 'ObjectHandling.c'))
    w.putln('%s = __Pyx_GetBuiltinName(%s); if (!%s) %s' % (cname, interned_cname, cname, w.error_goto(pos)))