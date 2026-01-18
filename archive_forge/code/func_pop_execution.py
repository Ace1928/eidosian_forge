from contextlib import contextmanager
from jedi import debug
from jedi.inference.base_value import NO_VALUES
def pop_execution(self):
    self._parent_execution_funcs.pop()
    self._recursion_level -= 1