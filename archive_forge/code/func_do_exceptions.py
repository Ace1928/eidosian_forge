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
def do_exceptions(self, arg):
    """exceptions [number]
            List or change current exception in an exception chain.
            Without arguments, list all the current exception in the exception
            chain. Exceptions will be numbered, with the current exception indicated
            with an arrow.
            If given an integer as argument, switch to the exception at that index.
            """
    if not self._chained_exceptions:
        self.message('Did not find chained exceptions. To move between exceptions, pdb/post_mortem must be given an exception object rather than a traceback.')
        return
    if not arg:
        for ix, exc in enumerate(self._chained_exceptions):
            prompt = '>' if ix == self._chained_exception_index else ' '
            rep = repr(exc)
            if len(rep) > 80:
                rep = rep[:77] + '...'
            indicator = '  -' if self._chained_exceptions[ix].__traceback__ is None else f'{ix:>3}'
            self.message(f'{prompt} {indicator} {rep}')
    else:
        try:
            number = int(arg)
        except ValueError:
            self.error('Argument must be an integer')
            return
        if 0 <= number < len(self._chained_exceptions):
            if self._chained_exceptions[number].__traceback__ is None:
                self.error('This exception does not have a traceback, cannot jump to it')
                return
            self._chained_exception_index = number
            self.setup(None, self._chained_exceptions[number].__traceback__)
            self.print_stack_entry(self.stack[self.curindex])
        else:
            self.error('No exception with that number')