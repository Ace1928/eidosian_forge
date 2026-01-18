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
def test_expression_serialization_arrow(pickle_module):
    for expr in create_sample_expressions()['all']:
        assert isinstance(expr, pc.Expression)
        restored = pickle_module.loads(pickle_module.dumps(expr))
        assert expr.equals(restored)