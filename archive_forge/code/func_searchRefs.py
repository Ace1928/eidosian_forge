from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
def searchRefs(obj, *args):
    """Pseudo-interactive function for tracing references backward.
    **Arguments:**
    
        obj:   The initial object from which to start searching
        args:  A set of string or int arguments.
               each integer selects one of obj's referrers to be the new 'obj'
               each string indicates an action to take on the current 'obj':
                  t:  print the types of obj's referrers
                  l:  print the lengths of obj's referrers (if they have __len__)
                  i:  print the IDs of obj's referrers
                  o:  print obj
                  ro: return obj
                  rr: return list of obj's referrers
    
    Examples::
    
       searchRefs(obj, 't')                    ## Print types of all objects referring to obj
       searchRefs(obj, 't', 0, 't')            ##   ..then select the first referrer and print the types of its referrers
       searchRefs(obj, 't', 0, 't', 'l')       ##   ..also print lengths of the last set of referrers
       searchRefs(obj, 0, 1, 'ro')             ## Select index 0 from obj's referrer, then select index 1 from the next set of referrers, then return that object
       
    """
    ignore = {id(sys._getframe()): None}
    gc.collect()
    refs = gc.get_referrers(obj)
    ignore[id(refs)] = None
    refs = [r for r in refs if id(r) not in ignore]
    for a in args:
        if type(a) is int:
            obj = refs[a]
            gc.collect()
            refs = gc.get_referrers(obj)
            ignore[id(refs)] = None
            refs = [r for r in refs if id(r) not in ignore]
        elif a == 't':
            print(list(map(typeStr, refs)))
        elif a == 'i':
            print(list(map(id, refs)))
        elif a == 'l':

            def slen(o):
                if hasattr(o, '__len__'):
                    return len(o)
                else:
                    return None
            print(list(map(slen, refs)))
        elif a == 'o':
            print(obj)
        elif a == 'ro':
            return obj
        elif a == 'rr':
            return refs