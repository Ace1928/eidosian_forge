import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def outputfile(fn):
    """This works out where to write test output.  If running
    code in a locked down file system, this will be a
    temp directory; otherwise, the output of 'test_foo.py' will
    normally be a file called 'test_foo.pdf', next door.
    """
    D = setOutDir(__name__)
    if fn:
        D = os.path.join(D, fn)
    return D