from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider
from _pydevd_bundle.pydevd_resolver import defaultResolver
from .pydevd_helpers import find_mod_attr
from _pydevd_bundle import pydevd_constants
import sys
def get_dictionary(self, obj):
    ret = dict()
    ret['__internals__'] = defaultResolver.get_dictionary(obj)
    if obj.size > 1024 * 1024:
        ret['min'] = 'ndarray too big, calculating min would slow down debugging'
        ret['max'] = 'ndarray too big, calculating max would slow down debugging'
    elif obj.size == 0:
        ret['min'] = 'array is empty'
        ret['max'] = 'array is empty'
    elif self.is_numeric(obj):
        ret['min'] = obj.min()
        ret['max'] = obj.max()
    else:
        ret['min'] = 'not a numeric object'
        ret['max'] = 'not a numeric object'
    ret['shape'] = obj.shape
    ret['dtype'] = obj.dtype
    ret['size'] = obj.size
    try:
        ret['[0:%s] ' % len(obj)] = list(obj[0:pydevd_constants.PYDEVD_CONTAINER_NUMPY_MAX_ITEMS])
    except:
        pass
    return ret