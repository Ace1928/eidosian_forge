import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
def decollate(self):
    """Packs Layout of DynamicMaps into a single DynamicMap that returns a Layout

        Decollation allows packing a Layout of DynamicMaps into a single DynamicMap
        that returns a Layout of simple (non-dynamic) elements. All nested streams are
        lifted to the resulting DynamicMap, and are available in the `streams`
        property.  The `callback` property of the resulting DynamicMap is a pure,
        stateless function of the stream values. To avoid stream parameter name
        conflicts, the resulting DynamicMap is configured with
        positional_stream_args=True, and the callback function accepts stream values
        as positional dict arguments.

        Returns:
            DynamicMap that returns a Layout
        """
    from .decollate import decollate
    return decollate(self)