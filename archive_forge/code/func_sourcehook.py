import os
import re
import sys
from collections import deque
from io import StringIO
def sourcehook(self, newfile):
    """Hook called on a filename to be sourced."""
    if newfile[0] == '"':
        newfile = newfile[1:-1]
    if isinstance(self.infile, str) and (not os.path.isabs(newfile)):
        newfile = os.path.join(os.path.dirname(self.infile), newfile)
    return (newfile, open(newfile, 'r'))