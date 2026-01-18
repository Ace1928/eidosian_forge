from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def init_handlers(self):
    """Create the default handlers."""
    self._handlers = {}
    self._esc_handlers = {}
    for handler in _default_handlers:
        handler(shell=self.shell, prefilter_manager=self, parent=self)