import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_multinomial_1d_pval():
    with pytest.raises(TypeError, match='pvals must be a 1-d'):
        random.multinomial(10, 0.3)