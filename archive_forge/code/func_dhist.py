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
@line_magic
def dhist(self, parameter_s=''):
    """Print your history of visited directories.

        %dhist       -> print full history\\
        %dhist n     -> print last n entries only\\
        %dhist n1 n2 -> print entries between n1 and n2 (n2 not included)\\

        This history is automatically maintained by the %cd command, and
        always available as the global list variable _dh. You can use %cd -<n>
        to go to directory number <n>.

        Note that most of time, you should view directory history by entering
        cd -<TAB>.

        """
    dh = self.shell.user_ns['_dh']
    if parameter_s:
        try:
            args = map(int, parameter_s.split())
        except:
            self.arg_err(self.dhist)
            return
        if len(args) == 1:
            ini, fin = (max(len(dh) - args[0], 0), len(dh))
        elif len(args) == 2:
            ini, fin = args
            fin = min(fin, len(dh))
        else:
            self.arg_err(self.dhist)
            return
    else:
        ini, fin = (0, len(dh))
    print('Directory history (kept in _dh)')
    for i in range(ini, fin):
        print('%d: %s' % (i, dh[i]))