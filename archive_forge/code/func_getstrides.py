import typing
from rpy2.rinterface_lib import openrlib
def getstrides(cdata, shape: typing.Tuple[int, ...], itemsize: int) -> typing.Tuple[int, ...]:
    """Get the strides (offsets in memory when walking along dimension)
    for an R array.

    The shape (see method `getshape`) and itemsize must be specified.
    Incorrect values are potentially unsage and result in a segfault.

    :param cdata: C data from cffi.
    :param shape: The shape of the array.
    :param itemsize: The size of (C sizeof) each item in the array.
    :return: A tuple with the strides. The length of the tuple is rank-1."""
    rk = len(shape)
    strides = [itemsize]
    for i in range(1, rk):
        strides.append(shape[i - 1] * strides[i - 1])
    return tuple(strides)