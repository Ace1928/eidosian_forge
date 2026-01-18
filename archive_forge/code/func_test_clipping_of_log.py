import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_clipping_of_log():
    path = Path._create_closed([(0.2, -99), (0.4, -99), (0.4, 20), (0.2, 20)])
    trans = mtransforms.BlendedGenericTransform(mtransforms.Affine2D(), scale.LogTransform(10, 'clip'))
    tpath = trans.transform_path_non_affine(path)
    result = tpath.iter_segments(trans.get_affine(), clip=(0, 0, 100, 100), simplify=False)
    tpoints, tcodes = zip(*result)
    assert_allclose(tcodes, path.codes[:-1])