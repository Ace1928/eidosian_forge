import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_copyright(self):
    """Test that all .py and .pyx files have a valid copyright statement"""
    incorrect = []
    copyright_re = re.compile('#\\s*copyright.*(?=\n)', re.I)
    copyright_statement_re = re.compile('# Copyright \\(C\\) (\\d+?)((, |-)\\d+)* [^ ]*')
    for fname, text in self.get_source_file_contents(extensions=('.py', '.pyx')):
        if self.is_copyright_exception(fname):
            continue
        match = copyright_statement_re.search(text)
        if not match:
            match = copyright_re.search(text)
            if match:
                incorrect.append((fname, 'found: {}'.format(match.group())))
            else:
                incorrect.append((fname, 'no copyright line found\n'))
        elif 'by Canonical' in match.group():
            incorrect.append((fname, 'should not have: "by Canonical": %s' % (match.group(),)))
    if incorrect:
        help_text = ['Some files have missing or incorrect copyright statements.', '', 'Please either add them to the list of COPYRIGHT_EXCEPTIONS in breezy/tests/test_source.py', "or add '# Copyright (C) 2007 Bazaar hackers' to these files:", '']
        for fname, comment in incorrect:
            help_text.append(fname)
            help_text.append(' ' * 4 + comment)
        self.fail('\n'.join(help_text))