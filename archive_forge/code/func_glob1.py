import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def glob1(dirname, pattern):
    return _glob1(dirname, pattern, None, False)