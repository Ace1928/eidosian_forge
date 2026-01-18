import operator
import warnings
import dask
from dask import core
from dask.core import istask
from dask.dataframe.core import _concat
from dask.dataframe.optimize import optimize
from dask.dataframe.shuffle import shuffle_group
from dask.highlevelgraph import HighLevelGraph
from .scheduler import MultipleReturnFunc, multiple_return_get
def rewrite_simple_shuffle_layer(dsk, keys):
    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
    else:
        dsk = dsk.copy()
    layers = dsk.layers.copy()
    for key, layer in layers.items():
        if type(layer) is SimpleShuffleLayer:
            dsk.layers[key] = MultipleReturnSimpleShuffleLayer.clone(layer)
    return dsk