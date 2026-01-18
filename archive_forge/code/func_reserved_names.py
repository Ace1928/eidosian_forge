import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
@util.memoized_property
def reserved_names(self):
    if self.enable_loop:
        return codegen.RESERVED_NAMES
    else:
        return codegen.RESERVED_NAMES.difference(['loop'])