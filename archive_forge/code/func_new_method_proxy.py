import copy
import itertools
import operator
from functools import wraps
def new_method_proxy(func):

    def inner(self, *args):
        if (_wrapped := self._wrapped) is empty:
            self._setup()
            _wrapped = self._wrapped
        return func(_wrapped, *args)
    inner._mask_wrapped = False
    return inner