from __future__ import annotations
from typing import Mapping, Union, Tuple
from collections.abc import Iterable, Mapping as _Mapping
from itertools import chain, islice
import numpy as np
from numpy.typing import ArrayLike
from qiskit.circuit import Parameter, QuantumCircuit
from .shape import ShapedMixin, ShapeInput, shape_tuple
def examine_array(*possible_shapes):
    nonlocal only_possible_shapes
    if only_possible_shapes is None:
        only_possible_shapes = set(possible_shapes)
    else:
        only_possible_shapes.intersection_update(possible_shapes)