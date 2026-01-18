from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider
from _pydevd_bundle.pydevd_resolver import defaultResolver
from .pydevd_helpers import find_mod_attr
from _pydevd_bundle import pydevd_constants
import sys
def can_provide(self, type_object, type_name):
    nd_array = find_mod_attr('numpy', 'ndarray')
    return nd_array is not None and issubclass(type_object, nd_array)