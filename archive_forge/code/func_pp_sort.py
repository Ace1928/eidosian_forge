import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_sort(self, s):
    if isinstance(s, z3.ArraySortRef):
        return seq1('Array', (self.pp_sort(s.domain()), self.pp_sort(s.range())))
    elif isinstance(s, z3.BitVecSortRef):
        return seq1('BitVec', (to_format(s.size()),))
    elif isinstance(s, z3.FPSortRef):
        return seq1('FPSort', (to_format(s.ebits()), to_format(s.sbits())))
    elif isinstance(s, z3.ReSortRef):
        return seq1('ReSort', (self.pp_sort(s.basis()),))
    elif isinstance(s, z3.SeqSortRef):
        if s.is_string():
            return to_format('String')
        return seq1('Seq', (self.pp_sort(s.basis()),))
    elif isinstance(s, z3.CharSortRef):
        return to_format('Char')
    else:
        return to_format(s.name())