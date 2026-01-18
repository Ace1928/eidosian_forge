import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def build_err_msg(arrays, err_msg, header='Items are not equal:', verbose=True, names=('ACTUAL', 'DESIRED'), precision=8):
    msg = ['\n' + header]
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            if isinstance(a, ndarray):
                r_func = ndarray.__repr__
            else:
                r_func = repr
            try:
                r = r_func(a)
            except Exception as exc:
                r = f'[repr failed for <{type(a).__name__}>: {exc}]'
            if r.count('\n') > 3:
                r = '\n'.join(r.splitlines()[:3])
                r += '...'
            msg.append(f' {names[i]}: {r}')
    return '\n'.join(msg)