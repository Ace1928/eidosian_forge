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
def cxOnePointLeafBiased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of choosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return (ind1, ind2)
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)
    common_types = set(types1.keys()).intersection(set(types2.keys()))
    if len(common_types) > 0:
        type_ = random.sample(common_types, 1)[0]
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = (ind2[slice2], ind1[slice1])
    return (ind1, ind2)