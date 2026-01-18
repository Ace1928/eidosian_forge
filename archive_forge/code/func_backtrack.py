import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def backtrack(self, steps=10, prob_type_1=0.3, prob_type_2=0.3):
    """
        Performs a sequence of Reidemeister moves which increase or maintain
        the number of crossings in a diagram.  The number of such
        moves is the parameter steps.  The diagram is modified in place.

        >>> K = Link('L14a7689')
        >>> K
        <Link L14a7689: 2 comp; 14 cross>
        >>> K.backtrack(steps = 5, prob_type_1 = 1, prob_type_2 = 0)
        >>> len(K.crossings)
        19
        >>> K.backtrack(steps = 5, prob_type_1 = 0, prob_type_2 = 1)
        >>> len(K.crossings)
        29
        """
    from . import simplify
    simplify.backtrack(self, num_steps=steps, prob_type_1=prob_type_1, prob_type_2=prob_type_2)