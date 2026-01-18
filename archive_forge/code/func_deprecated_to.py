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
def deprecated_to(version):
    """Context manager to expect DeprecationWarnings until a given version"""
    from packaging.version import Version
    from nibabel import __version__ as nbver
    if Version(nbver) < Version(version):
        return pytest.deprecated_call()
    return nullcontext()