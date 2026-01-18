import copy
from collections import OrderedDict
from math import log2
import numpy as np
from .. import functions as fn
def checkOverconstraint(self):
    """Check whether the system is overconstrained. If so, return the name of
        the first overconstrained parameter.

        Overconstraints occur when any fixed parameter can be successfully computed by the system.
        (Ideally, all parameters are either fixed by the user or constrained by the
        system, but never both).
        """
    for k, v in self._vars.items():
        if v[2] == 'fixed' and 'n' in v[3]:
            oldval = v[:]
            self.set(k, None, None)
            try:
                self.get(k)
                return k
            except RuntimeError:
                pass
            finally:
                self._vars[k] = oldval
    return False