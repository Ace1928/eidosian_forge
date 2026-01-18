import numpy as np
import pytest
from pandas import (
def test_is_monotonic_increasing_lexsorted(lexsorted_two_level_string_multiindex):
    mi = lexsorted_two_level_string_multiindex
    assert mi.is_monotonic_increasing is False
    assert Index(mi.values).is_monotonic_increasing is False
    assert mi._is_strictly_monotonic_increasing is False
    assert Index(mi.values)._is_strictly_monotonic_increasing is False