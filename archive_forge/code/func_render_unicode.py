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
def render_unicode(self, *args, **data):
    """Render the output of this template as a unicode object."""
    return runtime._render(self, self.callable_, args, data, as_unicode=True)