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
def test_generated_docstrings():
    assert pc.min_max.__doc__ == textwrap.dedent('        Compute the minimum and maximum values of a numeric array.\n\n        Null values are ignored by default.\n        This can be changed through ScalarAggregateOptions.\n\n        Parameters\n        ----------\n        array : Array-like\n            Argument to compute function.\n        skip_nulls : bool, default True\n            Whether to skip (ignore) nulls in the input.\n            If False, any null in the input forces the output to null.\n        min_count : int, default 1\n            Minimum number of non-null values in the input.  If the number\n            of non-null values is below `min_count`, the output is null.\n        options : pyarrow.compute.ScalarAggregateOptions, optional\n            Alternative way of passing options.\n        memory_pool : pyarrow.MemoryPool, optional\n            If not passed, will allocate memory from the default memory pool.\n        ')
    assert pc.add.__doc__ == textwrap.dedent('        Add the arguments element-wise.\n\n        Results will wrap around on integer overflow.\n        Use function "add_checked" if you want overflow\n        to return an error.\n\n        Parameters\n        ----------\n        x : Array-like or scalar-like\n            Argument to compute function.\n        y : Array-like or scalar-like\n            Argument to compute function.\n        memory_pool : pyarrow.MemoryPool, optional\n            If not passed, will allocate memory from the default memory pool.\n        ')
    assert pc.min_element_wise.__doc__ == textwrap.dedent('        Find the element-wise minimum value.\n\n        Nulls are ignored (by default) or propagated.\n        NaN is preferred over null, but not over any valid value.\n\n        Parameters\n        ----------\n        *args : Array-like or scalar-like\n            Argument to compute function.\n        skip_nulls : bool, default True\n            Whether to skip (ignore) nulls in the input.\n            If False, any null in the input forces the output to null.\n        options : pyarrow.compute.ElementWiseAggregateOptions, optional\n            Alternative way of passing options.\n        memory_pool : pyarrow.MemoryPool, optional\n            If not passed, will allocate memory from the default memory pool.\n        ')
    assert pc.filter.__doc__ == textwrap.dedent('        Filter with a boolean selection filter.\n\n        The output is populated with values from the input at positions\n        where the selection filter is non-zero.  Nulls in the selection filter\n        are handled based on FilterOptions.\n\n        Parameters\n        ----------\n        input : Array-like or scalar-like\n            Argument to compute function.\n        selection_filter : Array-like or scalar-like\n            Argument to compute function.\n        null_selection_behavior : str, default "drop"\n            How to handle nulls in the selection filter.\n            Accepted values are "drop", "emit_null".\n        options : pyarrow.compute.FilterOptions, optional\n            Alternative way of passing options.\n        memory_pool : pyarrow.MemoryPool, optional\n            If not passed, will allocate memory from the default memory pool.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> arr = pa.array(["a", "b", "c", None, "e"])\n        >>> mask = pa.array([True, False, None, False, True])\n        >>> arr.filter(mask)\n        <pyarrow.lib.StringArray object at ...>\n        [\n          "a",\n          "e"\n        ]\n        >>> arr.filter(mask, null_selection_behavior=\'emit_null\')\n        <pyarrow.lib.StringArray object at ...>\n        [\n          "a",\n          null,\n          "e"\n        ]\n        ')