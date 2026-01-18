from ._internal import NDArrayBase
from ..base import _Null
def dgl_subgraph(*data, **kwargs):
    """This operator constructs an induced subgraph for
    a given set of vertices from a graph. The operator accepts multiple
    sets of vertices as input. For each set of vertices, it returns a pair
    of CSR matrices if return_mapping is True: the first matrix contains edges
    with new edge Ids, the second matrix contains edges with the original
    edge Ids.

    Example:

       .. code:: python

         x=[[1, 0, 0, 2],
           [3, 0, 4, 0],
           [0, 5, 0, 0],
           [0, 6, 7, 0]]
         v = [0, 1, 2]
         dgl_subgraph(x, v, return_mapping=True) =
           [[1, 0, 0],
            [2, 0, 3],
            [0, 4, 0]],
           [[1, 0, 0],
            [3, 0, 4],
            [0, 5, 0]]



    Defined in ../src/operator/contrib/dgl_graph.cc:L1171

    Parameters
    ----------
    graph : NDArray
        Input graph where we sample vertices.
    data : NDArray[]
        The input arrays that include data arrays and states.
    return_mapping : boolean, required
        Return mapping of vid and eid between the subgraph and the parent graph.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)