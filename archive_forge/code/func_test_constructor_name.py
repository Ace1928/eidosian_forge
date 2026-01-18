from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_constructor_name(self):
    orig = RangeIndex(10)
    orig.name = 'original'
    copy = RangeIndex(orig)
    copy.name = 'copy'
    assert orig.name == 'original'
    assert copy.name == 'copy'
    new = Index(copy)
    assert new.name == 'copy'
    new.name = 'new'
    assert orig.name == 'original'
    assert copy.name == 'copy'
    assert new.name == 'new'