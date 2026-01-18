import os
import pkgutil
import re
import sys
import warnings
from functools import partial
from inspect import isgenerator, signature
from itertools import chain, product
from pathlib import Path
import numpy as np
import pytest
import sklearn
from sklearn.cluster import (
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import (
from sklearn.neighbors import (
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import _IS_WASM, IS_PYPY, all_estimators
from sklearn.utils._tags import _DEFAULT_TAGS, _safe_tags
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
@pytest.mark.parametrize('Estimator', [AffinityPropagation, Birch, MeanShift, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, LabelPropagation, LabelSpreading, OPTICS, SpectralClustering, LocalOutlierFactor, LocallyLinearEmbedding, Isomap, TSNE])
def test_f_contiguous_array_estimator(Estimator):
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=0)
    X = np.asfortranarray(X)
    y = np.round(X[:, 0])
    est = Estimator()
    est.fit(X, y)
    if hasattr(est, 'transform'):
        est.transform(X)
    if hasattr(est, 'predict'):
        est.predict(X)