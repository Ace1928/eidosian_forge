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
@embedded_active.setter
def embedded_active(self, value):
    if value:
        InteractiveShellEmbed._inactive_locations.discard(self._call_location_id)
        InteractiveShellEmbed._inactive_locations.discard(self._init_location_id)
    else:
        InteractiveShellEmbed._inactive_locations.add(self._call_location_id)