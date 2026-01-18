import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
@classmethod
def pprint_styles(cls):
    """Return the available styles as pretty-printed string."""
    table = [('Class', 'Name', 'Attrs'), *[(cls.__name__, f'``{name}``', str(inspect.signature(cls))[1:-1] or 'None') for name, cls in cls._style_list.items()]]
    col_len = [max((len(cell) for cell in column)) for column in zip(*table)]
    table_formatstr = '  '.join(('=' * cl for cl in col_len))
    rst_table = '\n'.join(['', table_formatstr, '  '.join((cell.ljust(cl) for cell, cl in zip(table[0], col_len))), table_formatstr, *['  '.join((cell.ljust(cl) for cell, cl in zip(row, col_len))) for row in table[1:]], table_formatstr])
    return textwrap.indent(rst_table, prefix=' ' * 4)