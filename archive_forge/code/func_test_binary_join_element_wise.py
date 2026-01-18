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
def test_binary_join_element_wise():
    null = pa.scalar(None, type=pa.string())
    arrs = [[None, 'a', 'b'], ['c', None, 'd'], [None, '-', '--']]
    assert pc.binary_join_element_wise(*arrs).to_pylist() == [None, None, 'b--d']
    assert pc.binary_join_element_wise('a', 'b', '-').as_py() == 'a-b'
    assert pc.binary_join_element_wise('a', null, '-').as_py() is None
    assert pc.binary_join_element_wise('a', 'b', null).as_py() is None
    skip = pc.JoinOptions(null_handling='skip')
    assert pc.binary_join_element_wise(*arrs, options=skip).to_pylist() == [None, 'a', 'b--d']
    assert pc.binary_join_element_wise('a', 'b', '-', options=skip).as_py() == 'a-b'
    assert pc.binary_join_element_wise('a', null, '-', options=skip).as_py() == 'a'
    assert pc.binary_join_element_wise('a', 'b', null, options=skip).as_py() is None
    replace = pc.JoinOptions(null_handling='replace', null_replacement='spam')
    assert pc.binary_join_element_wise(*arrs, options=replace).to_pylist() == [None, 'a-spam', 'b--d']
    assert pc.binary_join_element_wise('a', 'b', '-', options=replace).as_py() == 'a-b'
    assert pc.binary_join_element_wise('a', null, '-', options=replace).as_py() == 'a-spam'
    assert pc.binary_join_element_wise('a', 'b', null, options=replace).as_py() is None