from ._internal import NDArrayBase
from ..base import _Null
def box_encode(samples=None, matches=None, anchors=None, refs=None, means=None, stds=None, out=None, name=None, **kwargs):
    """Encode bounding boxes training target with normalized center offsets.
        Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.) array


    Defined in ../src/operator/contrib/bounding_box.cc:L210

    Parameters
    ----------
    samples : NDArray
        (B, N) value +1 (positive), -1 (negative), 0 (ignore)
    matches : NDArray
        (B, N) value range [0, M)
    anchors : NDArray
        (B, N, 4) encoded in corner
    refs : NDArray
        (B, M, 4) encoded in corner
    means : NDArray
        (4,) Mean value to be subtracted from encoded values
    stds : NDArray
        (4,) Std value to be divided from encoded values

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)