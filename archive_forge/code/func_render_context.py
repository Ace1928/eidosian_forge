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
def render_context(self, context, *args, **kwargs):
    """Render this :class:`.Template` with the given context.

        The data is written to the context's buffer.

        """
    if getattr(context, '_with_template', None) is None:
        context._set_with_template(self)
    runtime._render_context(self, self.callable_, context, *args, **kwargs)