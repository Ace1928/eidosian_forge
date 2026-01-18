from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def check_logical(self):
    """Build a line from tokens and run all logical checks on it."""
    self.report.increment_logical_line()
    mapping = self.build_tokens_line()
    if not mapping:
        return
    start_row, start_col = mapping[0][1]
    start_line = self.lines[start_row - 1]
    self.indent_level = expand_indent(start_line[:start_col])
    if self.blank_before < self.blank_lines:
        self.blank_before = self.blank_lines
    if self.verbose >= 2:
        print(self.logical_line[:80].rstrip())
    for name, check, argument_names in self._logical_checks:
        if self.verbose >= 4:
            print('   ' + name)
        self.init_checker_state(name, argument_names)
        for offset, text in self.run_check(check, argument_names) or ():
            if not isinstance(offset, tuple):
                for token_offset, pos in mapping:
                    if offset <= token_offset:
                        break
                offset = (pos[0], pos[1] + offset - token_offset)
            self.report_error(offset[0], offset[1], text, check)
    if self.logical_line:
        self.previous_indent_level = self.indent_level
        self.previous_logical = self.logical_line
        if not self.indent_level:
            self.previous_unindented_logical_line = self.logical_line
    self.blank_lines = 0
    self.tokens = []