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
def set_error_info(self, pos, used=False):
    self.funcstate.should_declare_error_indicator = True
    if used:
        self.funcstate.uses_error_indicator = True
    return '__PYX_MARK_ERR_POS(%s, %s)' % (self.lookup_filename(pos[0]), pos[1])