import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None