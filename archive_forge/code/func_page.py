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
def page(self, parameter_s=''):
    """Pretty print the object and display it through a pager.

        %page [options] OBJECT

        If no object is given, use _ (last output).

        Options:

          -r: page str(object), don't pretty-print it."""
    opts, args = self.parse_options(parameter_s, 'r')
    raw = 'r' in opts
    oname = args and args or '_'
    info = self.shell._ofind(oname)
    if info.found:
        if raw:
            txt = str(info.obj)
        else:
            txt = pformat(info.obj)
        page.page(txt)
    else:
        print('Object `%s` not found' % oname)