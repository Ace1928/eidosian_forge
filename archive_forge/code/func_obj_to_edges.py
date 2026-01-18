import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def obj_to_edges(obj) -> list:
    t = obj['object_type']
    if '_edge' in t:
        return [obj]
    elif t == 'line':
        return [line_to_edge(obj)]
    else:
        return {'rect': rect_to_edges, 'curve': curve_to_edges}[t](obj)