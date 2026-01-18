from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from contextlib import closing
import copy
import inspect
import os
import re
import sys
import textwrap
from io import StringIO
import numba.core.dispatcher
from numba.core import ir
def add_ir_line(func_data, line):
    line_str = line.strip()
    line_type = ''
    if line_str.endswith('pyobject'):
        line_str = line_str.replace('pyobject', '')
        line_type = 'pyobject'
    func_data['ir_lines'][num].append((line_str, line_type))
    indent_len = len(_getindent(line))
    func_data['ir_indent'][num].append(indent_len)