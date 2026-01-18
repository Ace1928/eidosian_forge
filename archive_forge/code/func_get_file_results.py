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
def get_file_results(self):
    """Print the result and return the overall count for this file."""
    self._deferred_print.sort()
    for line_number, offset, code, text, doc in self._deferred_print:
        print(self._fmt % {'path': self.filename, 'row': self.line_offset + line_number, 'col': offset + 1, 'code': code, 'text': text})
        if self._show_source:
            if line_number > len(self.lines):
                line = ''
            else:
                line = self.lines[line_number - 1]
            print(line.rstrip())
            print(re.sub('\\S', ' ', line[:offset]) + '^')
        if self._show_pep8 and doc:
            print('    ' + doc.strip())
        sys.stdout.flush()
    return self.file_errors