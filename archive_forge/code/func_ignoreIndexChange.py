import sys
from collections import OrderedDict
from ..Qt import QtWidgets
def ignoreIndexChange(func):

    def fn(self, *args, **kwds):
        prev = self._ignoreIndexChange
        self._ignoreIndexChange = True
        try:
            ret = func(self, *args, **kwds)
        finally:
            self._ignoreIndexChange = prev
        return ret
    return fn