import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def merge_hierarchical_mean_color(labels, rag, thresh, rag_copy=True, in_place_merge=False):
    return graph.merge_hierarchical(labels, rag, thresh, rag_copy, in_place_merge, _pre_merge_mean_color, _weight_mean_color)