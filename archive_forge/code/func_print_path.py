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
def print_path(path):
    for i, step in enumerate(path):
        next = path[(i + 1) % len(path)]
        outstream.write('   %s -- ' % type(step))
        if isinstance(step, dict):
            for key, val in step.items():
                if val is next:
                    outstream.write(f'[{key!r}]')
                    break
                if key is next:
                    outstream.write(f'[key] = {val!r}')
                    break
        elif isinstance(step, list):
            outstream.write('[%d]' % step.index(next))
        elif isinstance(step, tuple):
            outstream.write('( tuple )')
        else:
            outstream.write(repr(step))
        outstream.write(' ->\n')
    outstream.write('\n')