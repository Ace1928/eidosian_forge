import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def make_compound_path(cls, *args):
    """
        Concatenate a list of `Path`\\s into a single `Path`, removing all `STOP`\\s.
        """
    if not args:
        return Path(np.empty([0, 2], dtype=np.float32))
    vertices = np.concatenate([path.vertices for path in args])
    codes = np.empty(len(vertices), dtype=cls.code_type)
    i = 0
    for path in args:
        size = len(path.vertices)
        if path.codes is None:
            if size:
                codes[i] = cls.MOVETO
                codes[i + 1:i + size] = cls.LINETO
        else:
            codes[i:i + size] = path.codes
        i += size
    not_stop_mask = codes != cls.STOP
    return cls(vertices[not_stop_mask], codes[not_stop_mask])