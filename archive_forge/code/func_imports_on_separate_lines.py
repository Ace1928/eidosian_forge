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
def imports_on_separate_lines(logical_line):
    """Place imports on separate lines.

    Okay: import os\\nimport sys
    E401: import sys, os

    Okay: from subprocess import Popen, PIPE
    Okay: from myclas import MyClass
    Okay: from foo.bar.yourclass import YourClass
    Okay: import myclass
    Okay: import foo.bar.yourclass
    """
    line = logical_line
    if line.startswith('import '):
        found = line.find(',')
        if -1 < found and ';' not in line[:found]:
            yield (found, 'E401 multiple imports on one line')