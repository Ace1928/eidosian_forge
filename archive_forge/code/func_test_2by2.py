import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
def test_2by2(self, Source):
    source = Source()
    e_img, _, e_extent = source.combined(-1, 50, 2, 1)
    assert e_extent == (-1, 1, 50, 51)
    imgs = [source.single_tile(-1, 50)[0], source.single_tile(0, 50)[0]]
    assert_array_equal(np.hstack(imgs), e_img)