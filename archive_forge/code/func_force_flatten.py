import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def force_flatten(self):
    names = []
    types = []
    bitsizes = []
    fldquals = []
    for name, type, bitsize, quals in self.enumfields():
        names.append(name)
        types.append(type)
        bitsizes.append(bitsize)
        fldquals.append(quals)
    self.fldnames = tuple(names)
    self.fldtypes = tuple(types)
    self.fldbitsize = tuple(bitsizes)
    self.fldquals = tuple(fldquals)