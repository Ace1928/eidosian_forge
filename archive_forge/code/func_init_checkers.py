from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def init_checkers(self):
    """Create the default checkers."""
    self._checkers = []
    for checker in _default_checkers:
        checker(shell=self.shell, prefilter_manager=self, parent=self)