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
def put_chunk(self, chunk, context=None):
    context = context or self.context
    if context:
        chunk = sub_tempita(chunk, context)
    chunk = textwrap.dedent(chunk)
    for line in chunk.splitlines():
        self._putln(line)