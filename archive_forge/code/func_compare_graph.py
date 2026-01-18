import contextlib
from functools import partial
from unittest import TestCase
from unittest.util import safe_repr
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..core import (
from ..core.options import Cycle, Options
from ..core.util import cast_array_to_int64, datetime_types, dt_to_int, is_float
from . import *  # noqa (All Elements need to support comparison)
@classmethod
def compare_graph(cls, el1, el2, msg='Graph'):
    cls.compare_dataset(el1, el2, msg)
    cls.compare_nodes(el1.nodes, el2.nodes, msg)
    if el1._edgepaths or el2._edgepaths:
        cls.compare_edgepaths(el1.edgepaths, el2.edgepaths, msg)