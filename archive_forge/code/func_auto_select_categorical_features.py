import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
def auto_select_categorical_features(X, threshold=10):
    """Make a feature mask of categorical features in X.

    Features with less than 10 unique values are considered categorical.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Dense array or sparse matrix.

    threshold : int
        Maximum number of unique values per feature to consider the feature
        to be categorical.

    Returns
    -------
    feature_mask : array of booleans of size {n_features, }
    """
    feature_mask = []
    for column in range(X.shape[1]):
        if sparse.issparse(X):
            indptr_start = X.indptr[column]
            indptr_end = X.indptr[column + 1]
            unique = np.unique(X.data[indptr_start:indptr_end])
        else:
            unique = np.unique(X[:, column])
        feature_mask.append(len(unique) <= threshold)
    return feature_mask