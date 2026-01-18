import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def move_object(obj, axis: str, value):
    assert axis in ('h', 'v')
    if axis == 'h':
        new_items = [('x0', obj['x0'] + value), ('x1', obj['x1'] + value)]
    if axis == 'v':
        new_items = [('top', obj['top'] + value), ('bottom', obj['bottom'] + value)]
        if 'doctop' in obj:
            new_items += [('doctop', obj['doctop'] + value)]
        if 'y0' in obj:
            new_items += [('y0', obj['y0'] - value), ('y1', obj['y1'] - value)]
    return obj.__class__(tuple(obj.items()) + tuple(new_items))