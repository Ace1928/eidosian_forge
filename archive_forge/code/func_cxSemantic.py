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
def cxSemantic(ind1, ind2, gen_func=genGrow, pset=None, min=2, max=6):
    """
    Implementation of the Semantic Crossover operator [Geometric semantic genetic programming, Moraglio et al., 2012]
    offspring1 = random_tree1 * ind1 + (1 - random_tree1) * ind2
    offspring2 = random_tree1 * ind2 + (1 - random_tree1) * ind1

    :param ind1: first parent
    :param ind2: second parent
    :param gen_func: function responsible for the generation of the random tree that will be used during the mutation
    :param pset: Primitive Set, which contains terminal and operands to be used during the evolution
    :param min: min depth of the random tree
    :param max: max depth of the random tree
    :return: offsprings

    The mutated offspring contains parents

        >>> import operator
        >>> def lf(x): return 1 / (1 + math.exp(-x));
        >>> pset = PrimitiveSet("main", 2)
        >>> pset.addPrimitive(operator.sub, 2)
        >>> pset.addTerminal(3)
        >>> pset.addPrimitive(lf, 1, name="lf")
        >>> pset.addPrimitive(operator.add, 2)
        >>> pset.addPrimitive(operator.mul, 2)
        >>> ind1 = genGrow(pset, 1, 3)
        >>> ind2 = genGrow(pset, 1, 3)
        >>> new_ind1, new_ind2 = cxSemantic(ind1, ind2, pset=pset, max=2)
        >>> ctr = sum([n.name == ind1[i].name for i, n in enumerate(new_ind1)])
        >>> ctr == len(ind1)
        True
        >>> ctr = sum([n.name == ind2[i].name for i, n in enumerate(new_ind2)])
        >>> ctr == len(ind2)
        True
    """
    for p in ['lf', 'mul', 'add', 'sub']:
        assert p in pset.mapping, "A '" + p + "' function is required in order to perform semantic crossover"
    tr = gen_func(pset, min, max)
    tr.insert(0, pset.mapping['lf'])
    new_ind1 = ind1
    new_ind1.insert(0, pset.mapping['mul'])
    new_ind1.insert(0, pset.mapping['add'])
    new_ind1.extend(tr)
    new_ind1.append(pset.mapping['mul'])
    new_ind1.append(pset.mapping['sub'])
    new_ind1.append(Terminal(1.0, False, object))
    new_ind1.extend(tr)
    new_ind1.extend(ind2)
    new_ind2 = ind2
    new_ind2.insert(0, pset.mapping['mul'])
    new_ind2.insert(0, pset.mapping['add'])
    new_ind2.extend(tr)
    new_ind2.append(pset.mapping['mul'])
    new_ind2.append(pset.mapping['sub'])
    new_ind2.append(Terminal(1.0, False, object))
    new_ind2.extend(tr)
    new_ind2.extend(ind1)
    return (new_ind1, new_ind2)