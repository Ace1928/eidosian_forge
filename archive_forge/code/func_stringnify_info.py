import os
import sys
import errno
import pickle
import warnings
import collections
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.parallel import world, barrier
def stringnify_info(info):
    """Return a stringnified version of the dict *info* that is
    ensured to be picklable.  Items with non-string keys or
    unpicklable values are dropped and a warning is issued."""
    stringnified = {}
    for k, v in info.items():
        if not isinstance(k, str):
            warnings.warn('Non-string info-dict key is not stored in ' + 'trajectory: ' + repr(k), UserWarning)
            continue
        try:
            s = pickle.dumps(v, protocol=0)
        except pickle.PicklingError:
            warnings.warn('Skipping not picklable info-dict item: ' + '"%s" (%s)' % (k, sys.exc_info()[1]), UserWarning)
        else:
            stringnified[k] = s
    return stringnified