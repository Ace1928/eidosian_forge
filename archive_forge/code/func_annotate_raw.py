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
def annotate_raw(self):
    """
        This returns "raw" annotation information i.e. it has no output format
        specific markup included.
        """
    python_source = SourceLines(self.func_id.func)
    ir_lines = self.prepare_annotations()
    line_nums = [num for num in python_source]
    lifted_lines = [l.get_source_location() for l in self.lifted]

    def add_ir_line(func_data, line):
        line_str = line.strip()
        line_type = ''
        if line_str.endswith('pyobject'):
            line_str = line_str.replace('pyobject', '')
            line_type = 'pyobject'
        func_data['ir_lines'][num].append((line_str, line_type))
        indent_len = len(_getindent(line))
        func_data['ir_indent'][num].append(indent_len)
    func_key = (self.func_id.filename + ':' + str(self.func_id.firstlineno + 1), self.signature)
    if self.lifted_from is not None and self.lifted_from[1]['num_lifted_loops'] > 0:
        func_data = self.lifted_from[1]
        for num in line_nums:
            if num not in ir_lines.keys():
                continue
            func_data['ir_lines'][num] = []
            func_data['ir_indent'][num] = []
            for line in ir_lines[num]:
                add_ir_line(func_data, line)
                if line.strip().endswith('pyobject'):
                    func_data['python_tags'][num] = 'object_tag'
                    func_data['python_tags'][self.lifted_from[0]] = 'object_tag'
        self.lifted_from[1]['num_lifted_loops'] -= 1
    elif func_key not in TypeAnnotation.func_data.keys():
        TypeAnnotation.func_data[func_key] = {}
        func_data = TypeAnnotation.func_data[func_key]
        for i, loop in enumerate(self.lifted):
            loop.lifted_from = (lifted_lines[i], func_data)
        func_data['num_lifted_loops'] = self.num_lifted_loops
        func_data['filename'] = self.filename
        func_data['funcname'] = self.func_id.func_name
        func_data['python_lines'] = []
        func_data['python_indent'] = {}
        func_data['python_tags'] = {}
        func_data['ir_lines'] = {}
        func_data['ir_indent'] = {}
        for num in line_nums:
            func_data['python_lines'].append((num, python_source[num].strip()))
            indent_len = len(_getindent(python_source[num]))
            func_data['python_indent'][num] = indent_len
            func_data['python_tags'][num] = ''
            func_data['ir_lines'][num] = []
            func_data['ir_indent'][num] = []
            for line in ir_lines[num]:
                add_ir_line(func_data, line)
                if num in lifted_lines:
                    func_data['python_tags'][num] = 'lifted_tag'
                elif line.strip().endswith('pyobject'):
                    func_data['python_tags'][num] = 'object_tag'
    return self.func_data