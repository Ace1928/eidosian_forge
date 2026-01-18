import builtins
import os
import sys
import types
def release_gone():
    itop, mp, src = (id_to_obj.pop, memo.pop, getrefcount)
    [(itop(id_), mp(id_)) for id_, obj in list(id_to_obj.items()) if src(obj) < 4]