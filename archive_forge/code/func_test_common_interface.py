from copy import deepcopy
import numpy as np
import pytest
import nibabel.cifti2.cifti2_axes as axes
from .test_cifti2io_axes import check_rewrite
def test_common_interface():
    """
    Tests the common interface for all custom created CIFTI-2 axes
    """
    for axis1, axis2 in zip(get_axes(), get_axes()):
        assert axis1 == axis2
        concatenated = axis1 + axis2
        assert axis1 != concatenated
        assert axis1 == concatenated[:axis1.size]
        if isinstance(axis1, axes.SeriesAxis):
            assert axis2 != concatenated[axis1.size:]
        else:
            assert axis2 == concatenated[axis1.size:]
        assert len(axis1) == axis1.size