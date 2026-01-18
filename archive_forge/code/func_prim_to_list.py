import deap
from copy import copy
def prim_to_list(prim, args):
    if isinstance(prim, deap.gp.Terminal):
        if prim.name in pset.context:
            return pset.context[prim.name]
        else:
            return prim.value
    return [prim.name] + args