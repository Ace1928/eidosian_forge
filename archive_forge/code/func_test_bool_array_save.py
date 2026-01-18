from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread, imsave, plugin_order
from skimage._shared import testing
from skimage._shared.testing import fetch, assert_stacklevel
import pytest
def test_bool_array_save(self):
    with NamedTemporaryFile(suffix='.png') as f:
        fname = f.name
    with pytest.warns(UserWarning, match='.* is a boolean image') as record:
        a = np.zeros((5, 5), bool)
        a[2, 2] = True
        imsave(fname, a)
    assert_stacklevel(record)