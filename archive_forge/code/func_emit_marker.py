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
def emit_marker(self):
    pos, trace = self.last_pos
    self.last_marked_pos = pos
    self.last_pos = None
    self._write_lines('\n')
    if self.code_config.emit_code_comments:
        self.indent()
        self._write_lines('/* %s */\n' % self._build_marker(pos))
    if trace and self.funcstate and self.funcstate.can_trace and self.globalstate.directives['linetrace']:
        self.indent()
        self._write_lines('__Pyx_TraceLine(%d,%d,%s)\n' % (pos[1], not self.funcstate.gil_owned, self.error_goto(pos)))