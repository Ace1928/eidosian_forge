import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_coding_style(self):
    """Check if bazaar code conforms to some coding style conventions.

        Generally we expect PEP8, but we do not generally strictly enforce
        this, and there are existing files that do not comply.  The 'pep8'
        tool, available separately, will check for more cases.

        This test only enforces conditions that are globally true at the
        moment, and that should cause a patch to be rejected: spaces rather
        than tabs, unix newlines, and a newline at the end of the file.
        """
    tabs = {}
    illegal_newlines = {}
    no_newline_at_eof = []
    for fname, text in self.get_source_file_contents(extensions=('.py', '.pyx')):
        if not self.is_our_code(fname):
            continue
        lines = text.splitlines(True)
        last_line_no = len(lines) - 1
        for line_no, line in enumerate(lines):
            if '\t' in line:
                self._push_file(tabs, fname, line_no)
            if not line.endswith('\n') or line.endswith('\r\n'):
                if line_no != last_line_no:
                    self._push_file(illegal_newlines, fname, line_no)
        if not lines[-1].endswith('\n'):
            no_newline_at_eof.append(fname)
    problems = []
    if tabs:
        problems.append(self._format_message(tabs, 'Tab characters were found in the following source files.\nThey should either be replaced by "\\t" or by spaces:'))
    if illegal_newlines:
        problems.append(self._format_message(illegal_newlines, 'Non-unix newlines were found in the following source files:'))
    if no_newline_at_eof:
        no_newline_at_eof.sort()
        problems.append("The following source files doesn't have a newline at the end:\n\n    %s" % '\n    '.join(no_newline_at_eof))
    if problems:
        self.fail('\n\n'.join(problems))