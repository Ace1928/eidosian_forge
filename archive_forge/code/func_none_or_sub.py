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
def none_or_sub(self, s, context):
    """
        Format a string in this utility code with context. If None, do nothing.
        """
    if s is None:
        return None
    return sub_tempita(s, context, self.file, self.name)