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
def test_expression_construction():
    zero = pc.scalar(0)
    one = pc.scalar(1)
    true = pc.scalar(True)
    false = pc.scalar(False)
    string = pc.scalar('string')
    field = pc.field('field')
    nested_mixed_types = pc.field(b'a', 1, 'b')
    nested_field = pc.field(('nested', 'field'))
    nested_field2 = pc.field('nested', 'field')
    zero | one == string
    ~true == false
    for typ in ('bool', pa.bool_()):
        field.cast(typ) == true
    field.isin([1, 2])
    nested_mixed_types.isin(['foo', 'bar'])
    nested_field.isin(['foo', 'bar'])
    nested_field2.isin(['foo', 'bar'])
    with pytest.raises(TypeError):
        field.isin(1)
    with pytest.raises(pa.ArrowInvalid):
        field != object()