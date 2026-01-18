import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def is_model_valid(model, *random_data) -> bool:
    """Allow models with a maximum of 10 degree tilt from the vertical"""
    tilt = abs(np.arccos(np.dot(model.params[1], [0, 0, 1])))
    return tilt <= 10 / 180 * np.pi