from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def load_build(self):
    stack = self.stack
    state = stack.pop()
    inst = stack[-1]
    setstate = getattr(inst, '__setstate__', None)
    if setstate is not None:
        setstate(state)
        return
    slotstate = None
    if isinstance(state, tuple) and len(state) == 2:
        state, slotstate = state
    if state:
        inst_dict = inst.__dict__
        intern = sys.intern
        for k, v in state.items():
            if type(k) is str:
                inst_dict[intern(k)] = v
            else:
                inst_dict[k] = v
    if slotstate:
        for k, v in slotstate.items():
            setattr(inst, k, v)