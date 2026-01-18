from unittest import SkipTest
import numpy as np
from holoviews import Area, Bivariate, Contours, Distribution, Image, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.stats import bivariate_kde, univariate_kde
def test_bivariate_kde_contours_filled(self):
    np.random.seed(1)
    bivariate = Bivariate(np.random.rand(100, 2))
    kde = bivariate_kde(bivariate, n_samples=100, x_range=(0, 1), y_range=(0, 1), contours=True, filled=True, levels=10)
    self.assertIsInstance(kde, Polygons)
    self.assertEqual(len(kde.data), 10)