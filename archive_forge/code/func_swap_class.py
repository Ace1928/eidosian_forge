from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
def swap_class(cls, v2_cls, v1_cls, use_v2):
    """Swaps in v2_cls or v1_cls depending on graph mode."""
    if cls == object:
        return cls
    if cls in (v2_cls, v1_cls):
        return v2_cls if use_v2 else v1_cls
    new_bases = []
    for base in cls.__bases__:
        if use_v2 and issubclass(base, v1_cls) or (not use_v2 and issubclass(base, v2_cls)):
            new_base = swap_class(base, v2_cls, v1_cls, use_v2)
        else:
            new_base = base
        new_bases.append(new_base)
    cls.__bases__ = tuple(new_bases)
    return cls