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
@classmethod
def load_cached(cls, utility_code_name, from_file=None, context=None, __cache={}):
    context_key = tuple(sorted(context.items())) if context else None
    assert hash(context_key) is not None
    key = (cls, from_file, utility_code_name, context_key)
    try:
        return __cache[key]
    except KeyError:
        pass
    code = __cache[key] = cls.load(utility_code_name, from_file, context=context)
    return code