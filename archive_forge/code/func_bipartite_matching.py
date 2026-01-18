from ._internal import NDArrayBase
from ..base import _Null
def bipartite_matching(data=None, is_ascend=_Null, threshold=_Null, topk=_Null, out=None, name=None, **kwargs):
    """Compute bipartite matching.
      The matching is performed on score matrix with shape [B, N, M]
      - B: batch_size
      - N: number of rows to match
      - M: number of columns as reference to be matched against.

      Returns:
      x : matched column indices. -1 indicating non-matched elements in rows.
      y : matched row indices.

      Note::

        Zero gradients are back-propagated in this op for now.

      Example::

        s = [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
        x, y = bipartite_matching(x, threshold=1e-12, is_ascend=False)
        x = [1, -1, 0]
        y = [2, 0]



    Defined in ../src/operator/contrib/bounding_box.cc:L182

    Parameters
    ----------
    data : NDArray
        The input
    is_ascend : boolean, optional, default=0
        Use ascend order for scores instead of descending. Please set threshold accordingly.
    threshold : float, required
        Ignore matching when score < thresh, if is_ascend=false, or ignore score > thresh, if is_ascend=true.
    topk : int, optional, default='-1'
        Limit the number of matches to topk, set -1 for no limit

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)