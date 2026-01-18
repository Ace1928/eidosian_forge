from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def filter_aliases(metavar, dest, prefix):
    if not metavar:
        return dest if dest and dest.startswith(prefix) else ''
    a = metavar.replace(',', '').split()
    return ' '.join((x for x in a if x.startswith(prefix)))