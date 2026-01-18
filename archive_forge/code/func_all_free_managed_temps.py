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
def all_free_managed_temps(self):
    """Return a list of (cname, type) tuples of refcount-managed Python
        objects that are not currently in use.  This is used by
        try-except and try-finally blocks to clean up temps in the
        error case.
        """
    return sorted([(cname, type) for (type, manage_ref), freelist in self.temps_free.items() if manage_ref for cname in freelist[0]])