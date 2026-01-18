import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def fresh_copy(self):
    """Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.

        This is required when subclassing Graph with networkx v2 and
        does not cause problems for v1. Here is more detail from
        the network migrating from 1.x to 2.x document::

            With the new GraphViews (SubGraph, ReversedGraph, etc)
            you can't assume that ``G.__class__()`` will create a new
            instance of the same graph type as ``G``. In fact, the
            call signature for ``__class__`` differs depending on
            whether ``G`` is a view or a base class. For v2.x you
            should use ``G.fresh_copy()`` to create a null graph of
            the correct type---ready to fill with nodes and edges.

        """
    return RAG()