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
def check_physical(self, line):
    """Run all physical checks on a raw input line."""
    self.physical_line = line
    for name, check, argument_names in self._physical_checks:
        self.init_checker_state(name, argument_names)
        result = self.run_check(check, argument_names)
        if result is not None:
            offset, text = result
            self.report_error(self.line_number, offset, text, check)
            if text[:4] == 'E101':
                self.indent_char = line[0]