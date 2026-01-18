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
def check_ast(self):
    """Build the file's AST and run all AST checks."""
    try:
        tree = compile(''.join(self.lines), '', 'exec', PyCF_ONLY_AST)
    except (ValueError, SyntaxError, TypeError):
        return self.report_invalid_syntax()
    for name, cls, __ in self._ast_checks:
        checker = cls(tree, self.filename)
        for lineno, offset, text, check in checker.run():
            if not self.lines or not noqa(self.lines[lineno - 1]):
                self.report_error(lineno, offset, text, check)