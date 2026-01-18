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
@skip_doctest
@line_magic
def sc(self, parameter_s=''):
    """Shell capture - run shell command and capture output (DEPRECATED use !).

        DEPRECATED. Suboptimal, retained for backwards compatibility.

        You should use the form 'var = !command' instead. Example:

         "%sc -l myfiles = ls ~" should now be written as

         "myfiles = !ls ~"

        myfiles.s, myfiles.l and myfiles.n still apply as documented
        below.

        --
        %sc [options] varname=command

        IPython will run the given command using commands.getoutput(), and
        will then update the user's interactive namespace with a variable
        called varname, containing the value of the call.  Your command can
        contain shell wildcards, pipes, etc.

        The '=' sign in the syntax is mandatory, and the variable name you
        supply must follow Python's standard conventions for valid names.

        (A special format without variable name exists for internal use)

        Options:

          -l: list output.  Split the output on newlines into a list before
          assigning it to the given variable.  By default the output is stored
          as a single string.

          -v: verbose.  Print the contents of the variable.

        In most cases you should not need to split as a list, because the
        returned value is a special type of string which can automatically
        provide its contents either as a list (split on newlines) or as a
        space-separated string.  These are convenient, respectively, either
        for sequential processing or to be passed to a shell command.

        For example::

            # Capture into variable a
            In [1]: sc a=ls *py

            # a is a string with embedded newlines
            In [2]: a
            Out[2]: 'setup.py\\nwin32_manual_post_install.py'

            # which can be seen as a list:
            In [3]: a.l
            Out[3]: ['setup.py', 'win32_manual_post_install.py']

            # or as a whitespace-separated string:
            In [4]: a.s
            Out[4]: 'setup.py win32_manual_post_install.py'

            # a.s is useful to pass as a single command line:
            In [5]: !wc -l $a.s
              146 setup.py
              130 win32_manual_post_install.py
              276 total

            # while the list form is useful to loop over:
            In [6]: for f in a.l:
               ...:      !wc -l $f
               ...:
            146 setup.py
            130 win32_manual_post_install.py

        Similarly, the lists returned by the -l option are also special, in
        the sense that you can equally invoke the .s attribute on them to
        automatically get a whitespace-separated string from their contents::

            In [7]: sc -l b=ls *py

            In [8]: b
            Out[8]: ['setup.py', 'win32_manual_post_install.py']

            In [9]: b.s
            Out[9]: 'setup.py win32_manual_post_install.py'

        In summary, both the lists and strings used for output capture have
        the following special attributes::

            .l (or .list) : value as list.
            .n (or .nlstr): value as newline-separated string.
            .s (or .spstr): value as space-separated string.
        """
    opts, args = self.parse_options(parameter_s, 'lv')
    try:
        var, _ = args.split('=', 1)
        var = var.strip()
        _, cmd = parameter_s.split('=', 1)
    except ValueError:
        var, cmd = ('', '')
    split = 'l' in opts
    out = self.shell.getoutput(cmd, split=split)
    if 'v' in opts:
        print('%s ==\n%s' % (var, pformat(out)))
    if var:
        self.shell.user_ns.update({var: out})
    else:
        return out