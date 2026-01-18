from ._internal import NDArrayBase
from ..base import _Null
def amp_multicast(*data, **kwargs):
    """Cast function used by AMP, that casts its inputs to the common widest type.

    It casts only between low precision float/FP32 and does not do anything for other types.



    Defined in ../src/operator/tensor/amp_cast.cc:L169

    Parameters
    ----------
    data : NDArray[]
        Weights
    num_outputs : int, required
        Number of input/output pairs to be casted to the widest type.
    cast_narrow : boolean, optional, default=0
        Whether to cast to the narrowest type

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)