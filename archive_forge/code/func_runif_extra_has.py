from __future__ import annotations
import os
import re
import sys
import typing as ty
import unittest
import warnings
from contextlib import nullcontext
from itertools import zip_longest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .helpers import assert_data_similar, bytesio_filemap, bytesio_round_trip
from .np_features import memmap_after_ufunc
def runif_extra_has(test_str):
    """Decorator checks to see if NIPY_EXTRA_TESTS env var contains test_str"""
    return unittest.skipUnless(test_str in EXTRA_SET, f'Skip {test_str} tests.')