import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def fbind(self, name, func, *largs):
    self.binded_func[name] = partial(func, *largs)
    return True