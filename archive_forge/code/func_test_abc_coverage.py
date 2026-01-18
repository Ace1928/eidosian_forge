import re
import numpy as np
import pytest
from pandas.core.dtypes import generic as gt
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('abctype', [e for e in gt.__dict__ if e.startswith('ABC')])
def test_abc_coverage(self, abctype):
    assert abctype in (e for e, _ in self.abc_pairs) or abctype in self.abc_subclasses