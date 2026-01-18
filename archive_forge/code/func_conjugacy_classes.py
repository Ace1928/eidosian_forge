from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
def conjugacy_classes(self):
    """Return the conjugacy classes of the group.

        Explanation
        ===========

        As described in the documentation for the .conjugacy_class() function,
        conjugacy is an equivalence relation on a group G which partitions the
        set of elements. This method returns a list of all these conjugacy
        classes of G.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricGroup
        >>> SymmetricGroup(3).conjugacy_classes()
        [{(2)}, {(0 1 2), (0 2 1)}, {(0 2), (1 2), (2)(0 1)}]

        """
    identity = _af_new(list(range(self.degree)))
    known_elements = {identity}
    classes = [known_elements.copy()]
    for x in self.generate():
        if x not in known_elements:
            new_class = self.conjugacy_class(x)
            classes.append(new_class)
            known_elements.update(new_class)
    return classes