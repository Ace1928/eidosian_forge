from __future__ import absolute_import
import os
import re
import unittest
import shlex
import sys
import tempfile
import textwrap
from io import open
from functools import partial
from .Compiler import Errors
from .CodeWriter import CodeWriter
from .Compiler.TreeFragment import TreeFragment, strip_common_indent
from .Compiler.Visitor import TreeVisitor, VisitorTransform
from .Compiler import TreePath
def create_c_file_validator(self):
    patterns, antipatterns = (self._c_patterns, self._c_antipatterns)

    def fail(pos, pattern, found, file_path):
        Errors.error(pos, "Pattern '%s' %s found in %s" % (pattern, 'was' if found else 'was not', file_path))

    def extract_section(file_path, content, start, end):
        if start:
            split = re.search(start, content)
            if split:
                content = content[split.end():]
            else:
                fail(self._module_pos, start, found=False, file_path=file_path)
        if end:
            split = re.search(end, content)
            if split:
                content = content[:split.start()]
            else:
                fail(self._module_pos, end, found=False, file_path=file_path)
        return content

    def validate_file_content(file_path, content):
        for pattern in patterns:
            start, end, pattern = _parse_pattern(pattern)
            section = extract_section(file_path, content, start, end)
            if not re.search(pattern, section):
                fail(self._module_pos, pattern, found=False, file_path=file_path)
        for antipattern in antipatterns:
            start, end, antipattern = _parse_pattern(antipattern)
            section = extract_section(file_path, content, start, end)
            if re.search(antipattern, section):
                fail(self._module_pos, antipattern, found=True, file_path=file_path)

    def validate_c_file(result):
        c_file = result.c_file
        if not (patterns or antipatterns):
            return result
        with open(c_file, encoding='utf8') as f:
            content = f.read()
        content = _strip_c_comments(content)
        validate_file_content(c_file, content)
        html_file = os.path.splitext(c_file)[0] + '.html'
        if os.path.exists(html_file) and os.path.getmtime(c_file) <= os.path.getmtime(html_file):
            with open(html_file, encoding='utf8') as f:
                content = f.read()
            content = _strip_cython_code_from_html(content)
            validate_file_content(html_file, content)
    return validate_c_file