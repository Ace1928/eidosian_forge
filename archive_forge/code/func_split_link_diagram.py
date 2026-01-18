import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def split_link_diagram(self, destroy_original=False):
    """
        Breaks the given link diagram into pieces, one for each connected
        component of the underlying 4-valent graph.

        >>> L = Link([(2,1,1,2), (4,3,3,4)], check_planarity=False)
        >>> L.split_link_diagram()
        [<Link: 1 comp; 1 cross>, <Link: 1 comp; 1 cross>]
        """
    link = self.copy() if not destroy_original else self
    return [type(self)(list(component), check_planarity=False) for component in link.digraph().weak_components()]