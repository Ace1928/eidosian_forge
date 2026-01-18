import sys
import warnings
from IPython.core import ultratb, compilerop
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import DummyMod, InteractiveShell
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ipapp import load_default_config
from traitlets import Bool, CBool, Unicode
from IPython.utils.io import ask_yes_no
from typing import Set
@line_magic
def exit_raise(self, parameter_s=''):
    """%exit_raise Make the current embedded kernel exit and raise and exception.

        This function sets an internal flag so that an embedded IPython will
        raise a `IPython.terminal.embed.KillEmbedded` Exception on exit, and then exit the current I. This is
        useful to permanently exit a loop that create IPython embed instance.
        """
    self.shell.should_raise = True
    self.shell.ask_exit()