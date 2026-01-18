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
def sub_tempita(s, context, file=None, name=None):
    """Run tempita on string s with given context."""
    if not s:
        return None
    if file:
        context['__name'] = '%s:%s' % (file, name)
    elif name:
        context['__name'] = name
    from ..Tempita import sub
    return sub(s, **context)