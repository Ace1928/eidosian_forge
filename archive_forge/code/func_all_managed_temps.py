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
def all_managed_temps(self):
    """Return a list of (cname, type) tuples of refcount-managed Python objects.
        """
    return [(cname, type) for cname, type, manage_ref, static in self.temps_allocated if manage_ref]