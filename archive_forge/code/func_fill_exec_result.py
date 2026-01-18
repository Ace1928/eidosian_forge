import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
def fill_exec_result(self, result):
    if self.exec_result is not None:
        self.exec_result.result = result