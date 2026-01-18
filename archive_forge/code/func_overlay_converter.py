import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
def overlay_converter(src: 'Converter', target: 'Converter') -> None:
    """Overlay a converter onto an other.

    :param src: source of additional conversion rules
    :type src: :class:`Converter`
    :param target: target. The conversion rules in the src will
                   be added to this object.
    :type target: :class:`Converter`
    """
    for k, v in src.py2rpy.registry.items():
        if k is object and v is _py2rpy:
            continue
        target._py2rpy.register(k, v)
    for k, v in src.rpy2py.registry.items():
        if k is object and v is _rpy2py:
            continue
        target._rpy2py.register(k, v)
    for k, v in src.rpy2py_nc_map.items():
        if k in target.rpy2py_nc_map:
            target.rpy2py_nc_map[k].update(v._map.copy(), default=v._default)
        else:
            target.rpy2py_nc_map[k] = NameClassMap(v._default, namemap=v._map.copy())