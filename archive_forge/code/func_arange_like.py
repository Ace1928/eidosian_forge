from ._internal import NDArrayBase
from ..base import _Null
def arange_like(data=None, start=_Null, step=_Null, repeat=_Null, ctx=_Null, axis=_Null, out=None, name=None, **kwargs):
    """Return an array with evenly spaced values. If axis is not given, the output will 
    have the same shape as the input array. Otherwise, the output will be a 1-D array with size of 
    the specified axis in input shape.

    Examples::

      x = [[0.14883883 0.7772398  0.94865847 0.7225052 ]
           [0.23729339 0.6112595  0.66538996 0.5132841 ]
           [0.30822644 0.9912457  0.15502319 0.7043658 ]]
           <NDArray 3x4 @cpu(0)>

      out = mx.nd.contrib.arange_like(x, start=0)

        [[ 0.  1.  2.  3.]
         [ 4.  5.  6.  7.]
         [ 8.  9. 10. 11.]]
         <NDArray 3x4 @cpu(0)>

      out = mx.nd.contrib.arange_like(x, start=0, axis=-1)

        [0. 1. 2. 3.]
        <NDArray 4 @cpu(0)>


    Parameters
    ----------
    data : NDArray
        The input
    start : double, optional, default=0
        Start of interval. The interval includes this value. The default start value is 0.
    step : double, optional, default=1
        Spacing between values.
    repeat : int, optional, default='1'
        The repeating time of all elements. E.g repeat=3, the element a will be repeated three times --> a, a, a.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    axis : int or None, optional, default='None'
        Arange elements according to the size of a certain axis of input array. The negative numbers are interpreted counting from the backward. If not provided, will arange elements according to the input shape.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)