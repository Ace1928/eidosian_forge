import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
@pytest.mark.parametrize('image, reference, channel_axis', [(image_rgb, template_rgb, -1), (image_rgb[:, :, 0], template_rgb[:, :, 0], None)])
def test_match_histograms(self, image, reference, channel_axis):
    """Assert that pdf of matched image is close to the reference's pdf for
        all channels and all values of matched"""
    matched = exposure.match_histograms(image, reference, channel_axis=channel_axis)
    matched_pdf = self._calculate_image_empirical_pdf(matched)
    reference_pdf = self._calculate_image_empirical_pdf(reference)
    for channel in range(len(matched_pdf)):
        reference_values, reference_quantiles = reference_pdf[channel]
        matched_values, matched_quantiles = matched_pdf[channel]
        for i, matched_value in enumerate(matched_values):
            closest_id = np.abs(reference_values - matched_value).argmin()
            assert_almost_equal(matched_quantiles[i], reference_quantiles[closest_id], decimal=1)