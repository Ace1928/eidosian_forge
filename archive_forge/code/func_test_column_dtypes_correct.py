import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_column_dtypes_correct():
    msg = 'mismatch with expected type,'
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    for col in COL_DTYPES:
        r = region[col]
        if col in OBJECT_COLUMNS:
            assert COL_DTYPES[col] == object
            continue
        t = type(np.ravel(r)[0])
        if np.issubdtype(t, np.floating):
            assert COL_DTYPES[col] == float, f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
        elif np.issubdtype(t, np.integer):
            assert COL_DTYPES[col] == int, f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
        else:
            assert False, f'{col} dtype {t} {msg} {COL_DTYPES[col]}'