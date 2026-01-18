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
def doctest_mode(self, parameter_s=''):
    """Toggle doctest mode on and off.

        This mode is intended to make IPython behave as much as possible like a
        plain Python shell, from the perspective of how its prompts, exceptions
        and output look.  This makes it easy to copy and paste parts of a
        session into doctests.  It does so by:

        - Changing the prompts to the classic ``>>>`` ones.
        - Changing the exception reporting mode to 'Plain'.
        - Disabling pretty-printing of output.

        Note that IPython also supports the pasting of code snippets that have
        leading '>>>' and '...' prompts in them.  This means that you can paste
        doctests from files or docstrings (even if they have leading
        whitespace), and the code will execute correctly.  You can then use
        '%history -t' to see the translated history; this will give you the
        input after removal of all the leading prompts and whitespace, which
        can be pasted back into an editor.

        With these features, you can switch into this mode easily whenever you
        need to do testing and changes to doctests, without having to leave
        your existing IPython session.
        """
    shell = self.shell
    meta = shell.meta
    disp_formatter = self.shell.display_formatter
    ptformatter = disp_formatter.formatters['text/plain']
    dstore = meta.setdefault('doctest_mode', Struct())
    save_dstore = dstore.setdefault
    mode = save_dstore('mode', False)
    save_dstore('rc_pprint', ptformatter.pprint)
    save_dstore('xmode', shell.InteractiveTB.mode)
    save_dstore('rc_separate_out', shell.separate_out)
    save_dstore('rc_separate_out2', shell.separate_out2)
    save_dstore('rc_separate_in', shell.separate_in)
    save_dstore('rc_active_types', disp_formatter.active_types)
    if not mode:
        shell.separate_in = ''
        shell.separate_out = ''
        shell.separate_out2 = ''
        ptformatter.pprint = False
        disp_formatter.active_types = ['text/plain']
        shell.magic('xmode Plain')
    else:
        shell.separate_in = dstore.rc_separate_in
        shell.separate_out = dstore.rc_separate_out
        shell.separate_out2 = dstore.rc_separate_out2
        ptformatter.pprint = dstore.rc_pprint
        disp_formatter.active_types = dstore.rc_active_types
        shell.magic('xmode ' + dstore.xmode)
    shell.switch_doctest_mode(not mode)
    dstore.mode = bool(not mode)
    mode_label = ['OFF', 'ON'][dstore.mode]
    print('Doctest mode is:', mode_label)