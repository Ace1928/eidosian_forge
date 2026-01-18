import warnings
from gi import PyGIDeprecationWarning
from gi.repository import GLib
from ..overrides import override
from ..module import get_introspection_module
@classmethod
def new_from_data(cls, data, colorspace, has_alpha, bits_per_sample, width, height, rowstride, destroy_fn=None, *destroy_fn_data):
    if destroy_fn is not None:
        w = PyGIDeprecationWarning('destroy_fn argument deprecated')
        warnings.warn(w)
    if destroy_fn_data:
        w = PyGIDeprecationWarning('destroy_fn_data argument deprecated')
        warnings.warn(w)
    data = GLib.Bytes.new(data)
    return cls.new_from_bytes(data, colorspace, has_alpha, bits_per_sample, width, height, rowstride)