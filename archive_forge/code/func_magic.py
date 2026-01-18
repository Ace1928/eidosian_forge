from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
@line_magic
def magic(self, parameter_s=''):
    """Print information about the magic function system.

        Supported formats: -latex, -brief, -rest
        """
    mode = ''
    try:
        mode = parameter_s.split()[0][1:]
    except IndexError:
        pass
    brief = mode == 'brief'
    rest = mode == 'rest'
    magic_docs = self._magic_docs(brief, rest)
    if mode == 'latex':
        print(self.format_latex(magic_docs))
        return
    else:
        magic_docs = format_screen(magic_docs)
    out = ["\nIPython's 'magic' functions\n===========================\n\nThe magic function system provides a series of functions which allow you to\ncontrol the behavior of IPython itself, plus a lot of system-type\nfeatures. There are two kinds of magics, line-oriented and cell-oriented.\n\nLine magics are prefixed with the % character and work much like OS\ncommand-line calls: they get as an argument the rest of the line, where\narguments are passed without parentheses or quotes.  For example, this will\ntime the given statement::\n\n        %timeit range(1000)\n\nCell magics are prefixed with a double %%, and they are functions that get as\nan argument not only the rest of the line, but also the lines below it in a\nseparate argument.  These magics are called with two arguments: the rest of the\ncall line and the body of the cell, consisting of the lines below the first.\nFor example::\n\n        %%timeit x = numpy.random.randn((100, 100))\n        numpy.linalg.svd(x)\n\nwill time the execution of the numpy svd routine, running the assignment of x\nas part of the setup phase, which is not timed.\n\nIn a line-oriented client (the terminal or Qt console IPython), starting a new\ninput with %% will automatically enter cell mode, and IPython will continue\nreading input until a blank line is given.  In the notebook, simply type the\nwhole cell as one entity, but keep in mind that the %% escape can only be at\nthe very start of the cell.\n\nNOTE: If you have 'automagic' enabled (via the command line option or with the\n%automagic function), you don't need to type in the % explicitly for line\nmagics; cell magics always require an explicit '%%' escape.  By default,\nIPython ships with automagic on, so you should only rarely need the % escape.\n\nExample: typing '%cd mydir' (without the quotes) changes your working directory\nto 'mydir', if it exists.\n\nFor a list of the available magic functions, use %lsmagic. For a description\nof any of them, type %magic_name?, e.g. '%cd?'.\n\nCurrently the magic system has the following functions:", magic_docs, 'Summary of magic functions (from %slsmagic):' % magic_escapes['line'], str(self.lsmagic())]
    page.page('\n'.join(out))