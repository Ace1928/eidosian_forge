import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def popen(cmd, mode='r', buffering=-1):
    if not isinstance(cmd, str):
        raise TypeError('invalid cmd type (%s, expected string)' % type(cmd))
    if mode not in ('r', 'w'):
        raise ValueError('invalid mode %r' % mode)
    if buffering == 0 or buffering is None:
        raise ValueError('popen() does not support unbuffered streams')
    import subprocess
    if mode == 'r':
        proc = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, bufsize=buffering)
        return _wrap_close(proc.stdout, proc)
    else:
        proc = subprocess.Popen(cmd, shell=True, text=True, stdin=subprocess.PIPE, bufsize=buffering)
        return _wrap_close(proc.stdin, proc)