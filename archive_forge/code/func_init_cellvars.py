import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
def init_cellvars(parent, result, code):
    closure_cells = dict()
    side_effects = parent.output.side_effects
    for name in code.co_cellvars:
        closure_cells[name] = side_effects.track_cell_new()
        if name in result:
            side_effects.store_cell(closure_cells[name], result.pop(name))
    return closure_cells