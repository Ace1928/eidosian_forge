import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def find_vfunc_conflict_in_bases(vfunc, bases):
    for klass in bases:
        if not hasattr(klass, '__info__') or not hasattr(klass.__info__, 'get_vfuncs'):
            continue
        vfuncs = klass.__info__.get_vfuncs()
        vfunc_name = vfunc.get_name()
        for v in vfuncs:
            if v.get_name() == vfunc_name and v != vfunc:
                return klass
        aklass = find_vfunc_conflict_in_bases(vfunc, klass.__bases__)
        if aklass is not None:
            return aklass
    return None