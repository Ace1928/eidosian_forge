import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def deconnect_sum(self, destroy_original=False):
    """
        Undoes all connect sums that are diagramatically obvious,
        i.e. those where there is a circle which meets the projection
        in two points.

        >>> K5a1 = [(9,7,0,6), (3,9,4,8), (1,5,2,4), (7,3,8,2), (5,1,6,0)]
        >>> K = Link(K5a1)
        >>> L = K.connected_sum(K); L
        <Link: 1 comp; 10 cross>
        >>> L.deconnect_sum()
        [<Link: 1 comp; 5 cross>, <Link: 1 comp; 5 cross>]
        """
    from . import simplify
    link = self.copy() if not destroy_original else self
    return simplify.deconnect_sum(link)