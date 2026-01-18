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
def commented_file_contents(self, source_desc):
    try:
        return self.input_file_contents[source_desc]
    except KeyError:
        pass
    source_file = source_desc.get_lines(encoding='ASCII', error_handling='ignore')
    try:
        F = [u' * ' + line.rstrip().replace(u'*/', u'*[inserted by cython to avoid comment closer]/').replace(u'/*', u'/[inserted by cython to avoid comment start]*') for line in source_file]
    finally:
        if hasattr(source_file, 'close'):
            source_file.close()
    if not F:
        F.append(u'')
    self.input_file_contents[source_desc] = F
    return F