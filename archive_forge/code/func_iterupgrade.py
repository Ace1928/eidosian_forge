import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def iterupgrade(self, value):
    self._checked = True
    if not hasattr(value, '__iter__'):
        value = (value,)
    _strict_call = self._strict_call
    try:
        for _m in value:
            _strict_call(_m)
    except ValueError:
        self._do_upgrade()
        self.iterupgrade(value)