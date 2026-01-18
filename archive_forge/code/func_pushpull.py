import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
def pushpull(self, key, value, out=None, priority=0):
    """ Performs push and pull a single value or a sequence of values from the store.

        This function is coalesced form of push and pull operations. This function returns
        immediately after adding an operator to the engine. Subsequent attempts to read
        from the `out` variable will be blocked until the pull operation completes.

        `value` is pushed to the kvstore server for the specified keys and the updated
        values are pulled from the server to `out`. If `out` is not specified the pulled
        values are written to `value`. The returned values are guaranteed to be the latest
        values in the store.

        pushpull with `RowSparseNDArray` is not supported for dist kvstore.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        value : NDArray, list of NDArray, or list of list of NDArray
            Values corresponding to the keys.

        out: NDArray or list of NDArray or list of list of NDArray, optional
            Outputs corresponding to the keys.

        priority : int, optional
            The priority of the operation.
            Higher priority operations are likely to be executed before other actions.

        Examples
        --------
        >>> # pushpull a single key-value pair
        >>> kv.pushpull('3', mx.nd.ones(shape)*8, out=a)
        >>> print a.asnumpy()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # aggregate the value and then pushpull
        >>> gpus = [mx.gpu(i) for i in range(4)]
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.pushpull('3', b, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

        >>> # pushpull a list of keys.
        >>> # single device
        >>> keys = ['4', '5', '6']
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pushpull(keys, [mx.nd.ones(shape)]*len(keys), out=b)
        >>> print b[1].asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        >>> # multiple devices:
        >>> keys = ['7', '8', '9']
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.pushpull(keys, b)
        >>> print b[1][1].asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

        """
    cvkeys, cvals, use_str_keys = _ctype_key_value(key, value)
    if out is not None:
        cokeys, couts, _ = _ctype_key_value(key, out)
    else:
        cokeys = cvkeys
        couts = cvals
    if use_str_keys:
        check_call(_LIB.MXKVStorePushPullEx(self.handle, mx_uint(len(cvkeys)), cvkeys, mx_uint(len(cokeys)), cokeys, cvals, couts, ctypes.c_int(priority)))
    else:
        check_call(_LIB.MXKVStorePushPull(self.handle, mx_uint(len(cvkeys)), cvkeys, mx_uint(len(cokeys)), cokeys, cvals, couts, ctypes.c_int(priority)))