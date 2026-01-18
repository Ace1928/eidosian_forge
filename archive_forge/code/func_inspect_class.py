import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
def inspect_class(class_):
    """Extract a list of (name, wsattr|wsproperty) for the given class_"""
    attributes = []
    for name, attr in inspect.getmembers(class_, iswsattr):
        if name.startswith('_'):
            continue
        if inspect.isroutine(attr):
            continue
        if isinstance(attr, (wsattr, wsproperty)):
            attrdef = attr
        else:
            if attr not in native_types and (inspect.isclass(attr) or isinstance(attr, (list, dict))):
                register_type(attr)
            attrdef = getattr(class_, '__wsattrclass__', wsattr)(attr)
        attrdef.key = name
        if attrdef.name is None:
            attrdef.name = name
        attrdef.complextype = weakref.ref(class_)
        attributes.append(attrdef)
        setattr(class_, name, attrdef)
    sort_attributes(class_, attributes)
    return attributes