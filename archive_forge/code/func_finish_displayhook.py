import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
def finish_displayhook(self):
    """Finish up all displayhook activities."""
    sys.stdout.write(self.shell.separate_out2)
    sys.stdout.flush()