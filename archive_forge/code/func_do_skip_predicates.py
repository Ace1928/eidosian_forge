import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def do_skip_predicates(self, args):
    """
        Turn on/off individual predicates as to whether a frame should be hidden/skip.

        The global option to skip (or not) hidden frames is set with skip_hidden

        To change the value of a predicate

            skip_predicates key [true|false]

        Call without arguments to see the current values.

        To permanently change the value of an option add the corresponding
        command to your ``~/.pdbrc`` file. If you are programmatically using the
        Pdb instance you can also change the ``default_predicates`` class
        attribute.
        """
    if not args.strip():
        print('current predicates:')
        for p, v in self._predicates.items():
            print('   ', p, ':', v)
        return
    type_value = args.strip().split(' ')
    if len(type_value) != 2:
        print(f'Usage: skip_predicates <type> <value>, with <type> one of {set(self._predicates.keys())}')
        return
    type_, value = type_value
    if type_ not in self._predicates:
        print(f'{type_!r} not in {set(self._predicates.keys())}')
        return
    if value.lower() not in ('true', 'yes', '1', 'no', 'false', '0'):
        print(f"{value!r} is invalid - use one of ('true', 'yes', '1', 'no', 'false', '0')")
        return
    self._predicates[type_] = value.lower() in ('true', 'yes', '1')
    if not any(self._predicates.values()):
        print('Warning, all predicates set to False, skip_hidden may not have any effects.')