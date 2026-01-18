import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimensioned_constant_label(self):
    label = 'label'
    view = Dimensioned('An example of arbitrary data', label=label)
    self.assertEqual(view.label, label)
    try:
        view.label = 'another label'
        raise AssertionError('Label should be a constant parameter.')
    except TypeError:
        pass