import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def strip_math(s):
    """
    Remove latex formatting from mathtext.

    Only handles fully math and fully non-math strings.
    """
    if len(s) >= 2 and s[0] == s[-1] == '$':
        s = s[1:-1]
        for tex, plain in [('\\times', 'x'), ('\\mathdefault', ''), ('\\rm', ''), ('\\cal', ''), ('\\tt', ''), ('\\it', ''), ('\\', ''), ('{', ''), ('}', '')]:
            s = s.replace(tex, plain)
    return s