import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def roundtrip_truncated(arr):
    f = BytesIO()
    format.write_array(f, arr)
    f2 = BytesIO(f.getvalue()[0:-1])
    arr2 = format.read_array(f2)
    return arr2