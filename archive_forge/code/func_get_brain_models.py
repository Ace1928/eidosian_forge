from copy import deepcopy
import numpy as np
import pytest
import nibabel.cifti2.cifti2_axes as axes
from .test_cifti2io_axes import check_rewrite
def get_brain_models():
    """
    Generates a set of practice BrainModelAxis axes

    Yields
    ------
    BrainModelAxis axis
    """
    mask = np.zeros(vol_shape)
    mask[0, 1, 2] = 1
    mask[0, 4, 2] = True
    mask[0, 4, 0] = True
    yield axes.BrainModelAxis.from_mask(mask, 'ThalamusRight', rand_affine)
    mask[0, 0, 0] = True
    yield axes.BrainModelAxis.from_mask(mask, affine=rand_affine)
    yield axes.BrainModelAxis.from_surface([0, 5, 10], 15, 'CortexLeft')
    yield axes.BrainModelAxis.from_surface([0, 5, 10, 13], 15)
    surface_mask = np.zeros(15, dtype='bool')
    surface_mask[[2, 9, 14]] = True
    yield axes.BrainModelAxis.from_mask(surface_mask, name='CortexRight')