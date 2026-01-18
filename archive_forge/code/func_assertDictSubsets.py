from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def assertDictSubsets(self, sets: Sequence[Mapping[_K, _V]], subsets: Sequence[Mapping[_K, _V]]) -> None:
    """
        For each pair of corresponding elements in C{sets} and C{subsets},
        assert that the element from C{subsets} is a subset of the element from
        C{sets}.
        """
    self.assertEqual(len(sets), len(subsets))
    for a, b in zip(sets, subsets):
        self.assertDictSubset(a, b)