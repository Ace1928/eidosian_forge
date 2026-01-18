import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def mutShrink(individual):
    """This operator shrinks the *individual* by choosing randomly a branch and
    replacing it with one of the branch's arguments (also randomly chosen).

    :param individual: The tree to be shrunk.
    :returns: A tuple of one tree.
    """
    if len(individual) < 3 or individual.height <= 1:
        return (individual,)
    iprims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, Primitive) and node.ret in node.args:
            iprims.append((i, node))
    if len(iprims) != 0:
        index, prim = random.choice(iprims)
        arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)
        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree
    return (individual,)