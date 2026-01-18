from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def sort_transformers(self):
    """Sort the transformers by priority.

        This must be called after the priority of a transformer is changed.
        The :meth:`register_transformer` method calls this automatically.
        """
    self._transformers.sort(key=lambda x: x.priority)