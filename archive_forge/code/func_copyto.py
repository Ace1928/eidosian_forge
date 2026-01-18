import itertools
import numpy
from cupy import _core
from cupy._core import _fusion_interface
from cupy._core import fusion
from cupy._sorting import search
from cupy_backends.cuda.api import runtime
def copyto(dst, src, casting='same_kind', where=None):
    """Copies values from one array to another with broadcasting.

    This function can be called for arrays on different devices. In this case,
    casting, ``where``, and broadcasting is not supported, and an exception is
    raised if these are used.

    Args:
        dst (cupy.ndarray): Target array.
        src (cupy.ndarray): Source array.
        casting (str): Casting rule. See :func:`numpy.can_cast` for detail.
        where (cupy.ndarray of bool): If specified, this array acts as a mask,
            and an element is copied only if the corresponding element of
            ``where`` is True.

    .. seealso:: :func:`numpy.copyto`

    """
    src_is_numpy_scalar = False
    src_type = type(src)
    src_is_python_scalar = src_type in (int, bool, float, complex, fusion._FusionVarScalar, _fusion_interface._ScalarProxy)
    if src_is_python_scalar:
        src_dtype = numpy.dtype(type(src))
        can_cast = numpy.can_cast(src, dst.dtype, casting)
    elif isinstance(src, numpy.ndarray) or numpy.isscalar(src):
        if src.size != 1:
            raise ValueError('non-scalar numpy.ndarray cannot be used for copyto')
        src_dtype = src.dtype
        can_cast = numpy.can_cast(src, dst.dtype, casting)
        src = src.item()
        src_is_numpy_scalar = True
    else:
        src_dtype = src.dtype
        can_cast = numpy.can_cast(src_dtype, dst.dtype, casting)
    if not can_cast:
        raise TypeError('Cannot cast %s to %s in %s casting mode' % (src_dtype, dst.dtype, casting))
    if fusion._is_fusing():
        if where is None:
            _core.elementwise_copy(src, dst)
        else:
            fusion._call_ufunc(search._where_ufunc, where, src, dst, dst)
        return
    if not src_is_python_scalar and (not src_is_numpy_scalar):
        if not all([s in (d, 1) for s, d in itertools.zip_longest(reversed(src.shape), reversed(dst.shape), fillvalue=1)]):
            raise ValueError(f'could not broadcast input array from shape {src.shape} into shape {dst.shape}')
        squeeze_ndim = src.ndim - dst.ndim
        if squeeze_ndim > 0:
            src = src.squeeze(tuple(range(squeeze_ndim)))
    if where is not None:
        _core.elementwise_copy(src, dst, _where=where)
        return
    if dst.size == 0:
        return
    if src_is_python_scalar or src_is_numpy_scalar:
        _core.elementwise_copy(src, dst)
        return
    if _can_memcpy(dst, src):
        dst.data.copy_from_async(src.data, src.nbytes)
        return
    device = dst.device
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        if src.device != device:
            src = src.copy()
        _core.elementwise_copy(src, dst)
    finally:
        runtime.setDevice(prev_device)