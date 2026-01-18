from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_exported_functions():
    functions = exported_functions
    assert len(functions) >= 10
    for func in functions:
        desc = func.__arrow_compute_function__
        if desc['options_required']:
            continue
        arity = desc['arity']
        if arity == 0:
            continue
        if arity is Ellipsis:
            args = [object()] * 3
        else:
            args = [object()] * arity
        with pytest.raises(TypeError, match="Got unexpected argument type <class 'object'> for compute function"):
            func(*args)