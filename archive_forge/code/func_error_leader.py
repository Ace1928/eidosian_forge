import os
import re
import sys
from collections import deque
from io import StringIO
def error_leader(self, infile=None, lineno=None):
    """Emit a C-compiler-like, Emacs-friendly error-message leader."""
    if infile is None:
        infile = self.infile
    if lineno is None:
        lineno = self.lineno
    return '"%s", line %d: ' % (infile, lineno)