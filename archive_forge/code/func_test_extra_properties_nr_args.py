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
def test_extra_properties_nr_args():
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_few_args,))[0]
        _ = region.too_few_args
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_many_args,))[0]
        _ = region.too_many_args