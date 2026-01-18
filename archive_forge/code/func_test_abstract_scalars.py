import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
def test_abstract_scalars(install_temp):
    import checks
    assert checks.is_integer(1)
    assert checks.is_integer(np.int8(1))
    assert checks.is_integer(np.uint64(1))