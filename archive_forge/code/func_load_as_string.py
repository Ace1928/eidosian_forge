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
def load_as_string(cls, util_code_name, from_file, **kwargs):
    """
        Load a utility code as a string. Returns (proto, implementation)
        """
    util = cls.load(util_code_name, from_file, **kwargs)
    proto, impl = (util.proto, util.impl)
    return (util.format_code(proto), util.format_code(impl))