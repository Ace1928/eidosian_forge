import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_from_image(self):
    dist = Distribution(Image(np.arange(5) * np.arange(5)[:, np.newaxis]), 'z')
    assert dist.range(0) == (0, 16)