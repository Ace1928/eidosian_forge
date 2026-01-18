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
def funccontext_property(func):
    name = func.__name__
    attribute_of = operator.attrgetter(name)

    def get(self):
        return attribute_of(self.funcstate)

    def set(self, value):
        setattr(self.funcstate, name, value)
    return property(get, set)