import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def looping_usecase2(rec):
    a = rec('a')
    b = rec('b')
    cum = rec('cum')
    for x in a:
        rec.mark('--outer loop top--')
        cum = cum + x
        z = x + x
        rec.mark('--inner loop entry #{count}--')
        for y in b:
            rec.mark('--inner loop top #{count}--')
            cum = cum + y
            rec.mark('--inner loop bottom #{count}--')
        rec.mark('--inner loop exit #{count}--')
        if cum:
            cum = y + z
        else:
            break
        rec.mark('--outer loop bottom #{count}--')
    else:
        rec.mark('--outer loop else--')
    rec.mark('--outer loop exit--')
    return cum