from ._internal import NDArrayBase
from ..base import _Null
def dgl_adjacency(data=None, out=None, name=None, **kwargs):
    """This operator converts a CSR matrix whose values are edge Ids
    to an adjacency matrix whose values are ones. The output CSR matrix always has
    the data value of float32.

    Example:

       .. code:: python

      x = [[ 1, 0, 0 ],
           [ 0, 2, 0 ],
           [ 0, 0, 3 ]]
      dgl_adjacency(x) =
          [[ 1, 0, 0 ],
           [ 0, 1, 0 ],
           [ 0, 0, 1 ]]



    Defined in ../src/operator/contrib/dgl_graph.cc:L1424

    Parameters
    ----------
    data : NDArray
        Input ndarray

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)