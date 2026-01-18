from types import FunctionType
import sys
def determineMetaclass(bases, explicit_mc=None):
    """Determine metaclass from 1+ bases and optional explicit __metaclass__"""
    meta = [getattr(b, '__class__', type(b)) for b in bases]
    if explicit_mc is not None:
        meta.append(explicit_mc)
    if len(meta) == 1:
        return meta[0]
    candidates = minimalBases(meta)
    if len(candidates) > 1:
        raise TypeError('Incompatible metatypes', bases)
    return candidates[0]