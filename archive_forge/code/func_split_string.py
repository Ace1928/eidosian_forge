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
def split_string(matchobj):
    content = matchobj.group(1).replace('"', '"')
    return ''.join(('"%s\\n"\n' % line if not line.endswith('\\') or line.endswith('\\\\') else '"%s"\n' % line[:-1] for line in content.splitlines()))