from __future__ import absolute_import, division, print_function
import copy
def getchange(self):
    return self.diff if self._change is None else self._change