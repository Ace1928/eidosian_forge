import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def annotated_references(obj):
    """
    Return known information about references held by the given object.

    Returns a mapping from referents to lists of descriptions.  Note that there
    may be more than one edge leading to any particular referent; hence the
    need for a list.  Descriptions are currently strings.

    """
    references: Dict[int, List[str]] = {}

    def add_reference(name, obj):
        references.setdefault(id(obj), []).append(name)

    def add_attrs(*attrs):
        for attr in attrs:
            if hasattr(obj, attr):
                add_reference(attr, getattr(obj, attr))

    def add_cell_references():
        try:
            add_attrs('cell_contents')
        except ValueError:
            pass

    def add_function_references():
        add_attrs('__defaults__', '__closure__', '__globals__', '__code__', '__name__', '__module__', '__doc____qualname__', '__annotations__', '__kwdefaults__')

    def add_sequence_references():
        for position, item in enumerate(obj):
            add_reference(f'[{position}]', item)

    def add_dict_references():
        for key, value in obj.items():
            add_reference('key', key)
            add_reference(f'[{repr(key)}]', value)

    def add_set_references():
        for elt in obj:
            add_reference('element', elt)

    def add_bound_method_references():
        add_attrs('__self__', '__func__', 'im_class')

    def add_weakref_references():
        if type(obj) is weakref.ref:
            referents = gc.get_referents(obj)
            if len(referents) == 1:
                target = referents[0]
                add_reference('__callback__', target)

    def add_frame_references():
        f_locals = obj.f_locals
        add_attrs('f_back', 'f_code', 'f_builtins', 'f_globals', 'f_trace', 'f_locals')
        if type(f_locals) is dict:
            for name, local in obj.f_locals.items():
                add_reference(f'local {name}', local)

    def add_getset_descriptor_references():
        add_attrs('__objclass__', '__name__', '__doc__')
    type_based_references = {tuple: add_sequence_references, list: add_sequence_references, dict: add_dict_references, set: add_set_references, frozenset: add_set_references, types.FunctionType: add_function_references, types.FrameType: add_frame_references, CellType: add_cell_references, types.MethodType: add_bound_method_references, weakref.ref: add_weakref_references, types.GetSetDescriptorType: add_getset_descriptor_references}
    for type_ in type(obj).__mro__:
        if type_ in type_based_references:
            type_based_references[type_]()
    add_attrs('__dict__', '__class__')
    if isinstance(obj, type):
        add_attrs('__mro__')
    return references