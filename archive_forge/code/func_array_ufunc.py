from __future__ import annotations
import operator
from typing import Any
import numpy as np
from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op
from pandas.core.dtypes.generic import ABCNDFrame
from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer
def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    from pandas.core.frame import DataFrame, Series
    from pandas.core.generic import NDFrame
    from pandas.core.internals import ArrayManager, BlockManager
    cls = type(self)
    kwargs = _standardize_out_kwarg(**kwargs)
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result
    no_defer = (np.ndarray.__array_ufunc__, cls.__array_ufunc__)
    for item in inputs:
        higher_priority = hasattr(item, '__array_priority__') and item.__array_priority__ > self.__array_priority__
        has_array_ufunc = hasattr(item, '__array_ufunc__') and type(item).__array_ufunc__ not in no_defer and (not isinstance(item, self._HANDLED_TYPES))
        if higher_priority or has_array_ufunc:
            return NotImplemented
    types = tuple((type(x) for x in inputs))
    alignable = [x for x, t in zip(inputs, types) if issubclass(t, NDFrame)]
    if len(alignable) > 1:
        set_types = set(types)
        if len(set_types) > 1 and {DataFrame, Series}.issubset(set_types):
            raise NotImplementedError(f'Cannot apply ufunc {ufunc} to mixed DataFrame and Series inputs.')
        axes = self.axes
        for obj in alignable[1:]:
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes)):
                axes[i] = ax1.union(ax2)
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, axes))
        inputs = tuple((x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x for x, t in zip(inputs, types)))
    else:
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes))
    if self.ndim == 1:
        names = [getattr(x, 'name') for x in inputs if hasattr(x, 'name')]
        name = names[0] if len(set(names)) == 1 else None
        reconstruct_kwargs = {'name': name}
    else:
        reconstruct_kwargs = {}

    def reconstruct(result):
        if ufunc.nout > 1:
            return tuple((_reconstruct(x) for x in result))
        return _reconstruct(result)

    def _reconstruct(result):
        if lib.is_scalar(result):
            return result
        if result.ndim != self.ndim:
            if method == 'outer':
                raise NotImplementedError
            return result
        if isinstance(result, (BlockManager, ArrayManager)):
            result = self._constructor_from_mgr(result, axes=result.axes)
        else:
            result = self._constructor(result, **reconstruct_axes, **reconstruct_kwargs, copy=False)
        if len(alignable) == 1:
            result = result.__finalize__(self)
        return result
    if 'out' in kwargs:
        result = dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        return reconstruct(result)
    if method == 'reduce':
        result = dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        inputs = tuple((np.asarray(x) for x in inputs))
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        inputs = tuple((extract_array(x, extract_numpy=True) for x in inputs))
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif method == '__call__' and (not kwargs):
        mgr = inputs[0]._mgr
        result = mgr.apply(getattr(ufunc, method))
    else:
        result = default_array_ufunc(inputs[0], ufunc, method, *inputs, **kwargs)
    result = reconstruct(result)
    return result