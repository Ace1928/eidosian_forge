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
def bbox_list(regionmask):
    """Extra property whose output shape is dependent on mask shape."""
    return [1] * regionmask.shape[1]