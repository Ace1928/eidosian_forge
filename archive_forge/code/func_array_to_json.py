import warnings
import numpy as np
import zlib
from traitlets import Undefined, TraitError
from ipywidgets import widget_serialization, Widget
def array_to_json(value, widget):
    """Array JSON serializer."""
    if value is None:
        return None
    if value is Undefined:
        raise TraitError('Cannot serialize undefined array!')
    if isinstance(value, np.ndarray):
        if str(value.dtype) in ('int64', 'uint64'):
            warnings.warn('Cannot serialize (u)int64 data, Javascript does not support it. Casting to (u)int32.')
            value = value.astype(str(value.dtype).replace('64', '32'), order='C')
        elif not value.flags['C_CONTIGUOUS']:
            value = np.ascontiguousarray(value)
    return {'shape': value.shape, 'dtype': str(value.dtype), 'buffer': memoryview(value)}