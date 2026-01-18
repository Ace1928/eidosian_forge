import gc
import re
import sys
from IPython.core import page
from IPython.core.error import StdinNotImplementedError, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.encoding import DEFAULT_ENCODING
from IPython.utils.openpy import read_py_file
from IPython.utils.path import get_py_filename
@line_magic
def reset_selective(self, parameter_s=''):
    """Resets the namespace by removing names defined by the user.

        Input/Output history are left around in case you need them.

        %reset_selective [-f] regex

        No action is taken if regex is not included

        Options
          -f : force reset without asking for confirmation.

        See Also
        --------
        reset : invoked as ``%reset``

        Examples
        --------
        We first fully reset the namespace so your output looks identical to
        this example for pedagogical reasons; in practice you do not need a
        full reset::

          In [1]: %reset -f

        Now, with a clean namespace we can make a few variables and use
        ``%reset_selective`` to only delete names that match our regexp::

          In [2]: a=1; b=2; c=3; b1m=4; b2m=5; b3m=6; b4m=7; b2s=8

          In [3]: who_ls
          Out[3]: ['a', 'b', 'b1m', 'b2m', 'b2s', 'b3m', 'b4m', 'c']

          In [4]: %reset_selective -f b[2-3]m

          In [5]: who_ls
          Out[5]: ['a', 'b', 'b1m', 'b2s', 'b4m', 'c']

          In [6]: %reset_selective -f d

          In [7]: who_ls
          Out[7]: ['a', 'b', 'b1m', 'b2s', 'b4m', 'c']

          In [8]: %reset_selective -f c

          In [9]: who_ls
          Out[9]: ['a', 'b', 'b1m', 'b2s', 'b4m']

          In [10]: %reset_selective -f b

          In [11]: who_ls
          Out[11]: ['a']

        Notes
        -----
        Calling this magic from clients that do not implement standard input,
        such as the ipython notebook interface, will reset the namespace
        without confirmation.
        """
    opts, regex = self.parse_options(parameter_s, 'f')
    if 'f' in opts:
        ans = True
    else:
        try:
            ans = self.shell.ask_yes_no('Once deleted, variables cannot be recovered. Proceed (y/[n])? ', default='n')
        except StdinNotImplementedError:
            ans = True
    if not ans:
        print('Nothing done.')
        return
    user_ns = self.shell.user_ns
    if not regex:
        print('No regex pattern specified. Nothing done.')
        return
    else:
        try:
            m = re.compile(regex)
        except TypeError as e:
            raise TypeError('regex must be a string or compiled pattern') from e
        for i in self.who_ls():
            if m.search(i):
                del user_ns[i]