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
def inject_unbound_methods(self, impl, output):
    """Replace 'UNBOUND_METHOD(type, "name")' by a constant Python identifier cname.
        """
    if 'CALL_UNBOUND_METHOD(' not in impl:
        return (False, impl)

    def externalise(matchobj):
        type_cname, method_name, obj_cname, args = matchobj.groups()
        args = [arg.strip() for arg in args[1:].split(',')] if args else []
        assert len(args) < 3, 'CALL_UNBOUND_METHOD() does not support %d call arguments' % len(args)
        return output.cached_unbound_method_call_code(obj_cname, type_cname, method_name, args)
    impl = re.sub('CALL_UNBOUND_METHOD\\(([a-zA-Z_]+),\\s*"([^"]+)",\\s*([^),]+)((?:,[^),]+)*)\\)', externalise, impl)
    assert 'CALL_UNBOUND_METHOD(' not in impl
    return (True, impl)