from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
@cython.locals(start=cython.Py_ssize_t, q=cython.Py_ssize_t, single_q=cython.Py_ssize_t, double_q=cython.Py_ssize_t, hash_mark=cython.Py_ssize_t, end=cython.Py_ssize_t, k=cython.Py_ssize_t, counter=cython.Py_ssize_t, quote_len=cython.Py_ssize_t)
def strip_string_literals(code, prefix='__Pyx_L'):
    """
    Normalizes every string literal to be of the form '__Pyx_Lxxx',
    returning the normalized code and a mapping of labels to
    string literals.
    """
    new_code = []
    literals = {}
    counter = 0
    start = q = 0
    in_quote = False
    hash_mark = single_q = double_q = -1
    code_len = len(code)
    quote_type = None
    quote_len = -1
    while True:
        if hash_mark < q:
            hash_mark = code.find('#', q)
        if single_q < q:
            single_q = code.find("'", q)
        if double_q < q:
            double_q = code.find('"', q)
        q = min(single_q, double_q)
        if q == -1:
            q = max(single_q, double_q)
        if q == -1 and hash_mark == -1:
            new_code.append(code[start:])
            break
        elif in_quote:
            if code[q - 1] == u'\\':
                k = 2
                while q >= k and code[q - k] == u'\\':
                    k += 1
                if k % 2 == 0:
                    q += 1
                    continue
            if code[q] == quote_type and (quote_len == 1 or (code_len > q + 2 and quote_type == code[q + 1] == code[q + 2])):
                counter += 1
                label = '%s%s_' % (prefix, counter)
                literals[label] = code[start + quote_len:q]
                full_quote = code[q:q + quote_len]
                new_code.append(full_quote)
                new_code.append(label)
                new_code.append(full_quote)
                q += quote_len
                in_quote = False
                start = q
            else:
                q += 1
        elif -1 != hash_mark and (hash_mark < q or q == -1):
            new_code.append(code[start:hash_mark + 1])
            end = code.find('\n', hash_mark)
            counter += 1
            label = '%s%s_' % (prefix, counter)
            if end == -1:
                end_or_none = None
            else:
                end_or_none = end
            literals[label] = code[hash_mark + 1:end_or_none]
            new_code.append(label)
            if end == -1:
                break
            start = q = end
        else:
            if code_len >= q + 3 and code[q] == code[q + 1] == code[q + 2]:
                quote_len = 3
            else:
                quote_len = 1
            in_quote = True
            quote_type = code[q]
            new_code.append(code[start:q])
            start = q
            q += quote_len
    return (''.join(new_code), literals)