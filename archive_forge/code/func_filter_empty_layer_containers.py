import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def filter_empty_layer_containers(layer_list):
    """Filter out empty Layer-like containers and uniquify."""
    existing = set()
    to_visit = layer_list[::-1]
    while to_visit:
        obj = to_visit.pop()
        if id(obj) in existing:
            continue
        existing.add(id(obj))
        if hasattr(obj, '_is_layer') and (not isinstance(obj, type)):
            yield obj
        else:
            sub_layers = getattr(obj, 'layers', None) or []
            to_visit.extend(sub_layers[::-1])