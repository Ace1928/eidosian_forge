import io
import os
import pathlib
import re
import sys
from pprint import pformat
from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn
@line_cell_magic
def sx(self, line='', cell=None):
    """Shell execute - run shell command and capture output (!! is short-hand).

        %sx command

        IPython will run the given command using commands.getoutput(), and
        return the result formatted as a list (split on '\\n').  Since the
        output is _returned_, it will be stored in ipython's regular output
        cache Out[N] and in the '_N' automatic variables.

        Notes:

        1) If an input line begins with '!!', then %sx is automatically
        invoked.  That is, while::

          !ls

        causes ipython to simply issue system('ls'), typing::

          !!ls

        is a shorthand equivalent to::

          %sx ls

        2) %sx differs from %sc in that %sx automatically splits into a list,
        like '%sc -l'.  The reason for this is to make it as easy as possible
        to process line-oriented shell output via further python commands.
        %sc is meant to provide much finer control, but requires more
        typing.

        3) Just like %sc -l, this is a list with special attributes:
        ::

          .l (or .list) : value as list.
          .n (or .nlstr): value as newline-separated string.
          .s (or .spstr): value as whitespace-separated string.

        This is very useful when trying to use such lists as arguments to
        system commands."""
    if cell is None:
        return self.shell.getoutput(line)
    else:
        opts, args = self.parse_options(line, '', 'out=')
        output = self.shell.getoutput(cell)
        out_name = opts.get('out', opts.get('o'))
        if out_name:
            self.shell.user_ns[out_name] = output
        else:
            return output