import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
def test_feature_detector():
    with pytest.raises(NotImplementedError):
        FeatureDetector().detect(None)