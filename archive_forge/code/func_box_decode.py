from ._internal import NDArrayBase
from ..base import _Null
def box_decode(data=None, anchors=None, std0=_Null, std1=_Null, std2=_Null, std3=_Null, clip=_Null, format=_Null, out=None, name=None, **kwargs):
    """Decode bounding boxes training target with normalized center offsets.
        Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`
        or center type: `x, y, width, height.) array


    Defined in ../src/operator/contrib/bounding_box.cc:L233

    Parameters
    ----------
    data : NDArray
        (B, N, 4) predicted bbox offset
    anchors : NDArray
        (1, N, 4) encoded in corner or center
    std0 : float, optional, default=1
        value to be divided from the 1st encoded values
    std1 : float, optional, default=1
        value to be divided from the 2nd encoded values
    std2 : float, optional, default=1
        value to be divided from the 3rd encoded values
    std3 : float, optional, default=1
        value to be divided from the 4th encoded values
    clip : float, optional, default=-1
        If larger than 0, bounding box target will be clipped to this value.
    format : {'center', 'corner'},optional, default='center'
        The box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)