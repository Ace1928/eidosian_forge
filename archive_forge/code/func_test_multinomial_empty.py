import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_multinomial_empty():
    assert random.multinomial(10, []).shape == (0,)
    assert random.multinomial(3, [], size=(7, 5, 3)).shape == (7, 5, 3, 0)