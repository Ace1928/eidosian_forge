import os
import sys
from pathlib import Path
import numpy as np
from numpy.testing import assert_
def test_isfile(self):
    """Test if all ``.pyi`` files are properly installed."""
    for file in FILES:
        assert_(os.path.isfile(file))