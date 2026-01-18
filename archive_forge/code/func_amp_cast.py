from ._internal import NDArrayBase
from ..base import _Null
def amp_cast(data=None, dtype=_Null, out=None, name=None, **kwargs):
    """Cast function between low precision float/FP32 used by AMP.

    It casts only between low precision float/FP32 and does not do anything for other types.


    Defined in ../src/operator/tensor/amp_cast.cc:L125

    Parameters
    ----------
    data : NDArray
        The input.
    dtype : {'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'}, required
        Output data type.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)