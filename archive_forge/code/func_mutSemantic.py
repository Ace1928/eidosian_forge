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
def mutSemantic(individual, gen_func=genGrow, pset=None, ms=None, min=2, max=6):
    """
    Implementation of the Semantic Mutation operator. [Geometric semantic genetic programming, Moraglio et al., 2012]
    mutated_individual = individual + logistic * (random_tree1 - random_tree2)

    :param individual: individual to mutate
    :param gen_func: function responsible for the generation of the random tree that will be used during the mutation
    :param pset: Primitive Set, which contains terminal and operands to be used during the evolution
    :param ms: Mutation Step
    :param min: min depth of the random tree
    :param max: max depth of the random tree
    :return: mutated individual

    The mutated contains the original individual

        >>> import operator
        >>> def lf(x): return 1 / (1 + math.exp(-x));
        >>> pset = PrimitiveSet("main", 2)
        >>> pset.addPrimitive(operator.sub, 2)
        >>> pset.addTerminal(3)
        >>> pset.addPrimitive(lf, 1, name="lf")
        >>> pset.addPrimitive(operator.add, 2)
        >>> pset.addPrimitive(operator.mul, 2)
        >>> individual = genGrow(pset, 1, 3)
        >>> mutated = mutSemantic(individual, pset=pset, max=2)
        >>> ctr = sum([m.name == individual[i].name for i, m in enumerate(mutated[0])])
        >>> ctr == len(individual)
        True
    """
    for p in ['lf', 'mul', 'add', 'sub']:
        assert p in pset.mapping, "A '" + p + "' function is required in order to perform semantic mutation"
    tr1 = gen_func(pset, min, max)
    tr2 = gen_func(pset, min, max)
    tr1.insert(0, pset.mapping['lf'])
    tr2.insert(0, pset.mapping['lf'])
    if ms is None:
        ms = random.uniform(0, 2)
    mutation_step = Terminal(ms, False, object)
    new_ind = individual
    new_ind.insert(0, pset.mapping['add'])
    new_ind.append(pset.mapping['mul'])
    new_ind.append(mutation_step)
    new_ind.append(pset.mapping['sub'])
    new_ind.extend(tr1)
    new_ind.extend(tr2)
    return (new_ind,)