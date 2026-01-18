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
@classmethod
def get_module_source_metadata(cls, module_source, full_line_map=False):
    source_map = re.search('__M_BEGIN_METADATA(.+?)__M_END_METADATA', module_source, re.S).group(1)
    source_map = json.loads(source_map)
    source_map['line_map'] = {int(k): int(v) for k, v in source_map['line_map'].items()}
    if full_line_map:
        f_line_map = source_map['full_line_map'] = []
        line_map = source_map['line_map']
        curr_templ_line = 1
        for mod_line in range(1, max(line_map)):
            if mod_line in line_map:
                curr_templ_line = line_map[mod_line]
            f_line_map.append(curr_templ_line)
    return source_map