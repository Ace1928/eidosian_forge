from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
def pca(data, n_components=100, eps=0.3, method='svd', seed=None, return_singular_values=False, n_pca=None, svd_offset=None, svd_multiples=None):
    """Calculate PCA using random projections to handle sparse matrices.

    Uses the Johnson-Lindenstrauss Lemma to determine the number of
    dimensions of random projections prior to subtracting the mean.
    Dense matrices are provided to `sklearn.decomposition.PCA` directly.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    n_components : int, optional (default: 100)
        Number of PCs to compute
    eps : strictly positive float, optional (default=0.3)
        Parameter to control the quality of the embedding of sparse input.
        Smaller values lead to more accurate embeddings but higher
        computational and memory costs
    method : {'svd', 'orth_rproj', 'rproj', 'dense'}, optional (default: 'svd')
        Dimensionality reduction method applied prior to mean centering
        of sparse input. The method choice affects accuracy
        (`svd` > `orth_rproj` > `rproj`) and comes with increased
        computational cost (but not memory.) On the other hand,
        `method='dense'` adds a memory cost but is faster.
    seed : int, RandomState or None, optional (default: None)
        Random state.
    return_singular_values : bool, optional (default: False)
        If True, also return the singular values
    n_pca : Deprecated.
    svd_offset : Deprecated.
    svd_multiples :Deprecated.

    Returns
    -------
    data_pca : array-like, shape=[n_samples, n_components]
        PCA reduction of `data`
    singular_values : list-like, shape=[n_components]
        Singular values corresponding to principal components
        returned only if return_values is True
    """
    if n_pca is not None:
        warnings.warn('n_pca is deprecated. Setting n_components={}.'.format(n_pca), FutureWarning)
        n_components = n_pca
    if svd_offset is not None:
        warnings.warn('svd_offset is deprecated. Please use `eps` instead.', FutureWarning)
    if svd_multiples is not None:
        warnings.warn('svd_multiples is deprecated. Please use `eps` instead.', FutureWarning)
    if not 0 < n_components <= min(data.shape):
        raise ValueError('n_components={} must be between 0 and min(n_samples, n_features)={}'.format(n_components, min(data.shape)))
    if isinstance(data, pd.DataFrame):
        index = data.index
    else:
        index = None
    if method == 'dense':
        data = utils.toarray(data)
    else:
        data = utils.to_array_or_spmatrix(data)
    if sparse.issparse(data):
        try:
            pca_op = SparseInputPCA(n_components=n_components, eps=eps, method=method, random_state=seed)
            data = pca_op.fit_transform(data)
        except RuntimeError as e:
            if 'which is larger than the original space' in str(e):
                return pca(utils.toarray(data), n_components=n_components, seed=seed, return_singular_values=return_singular_values)
    else:
        pca_op = decomposition.PCA(n_components, random_state=seed)
        data = pca_op.fit_transform(data)
    if index is not None:
        data = pd.DataFrame(data, index=index, columns=['PC{}'.format(i + 1) for i in range(n_components)])
    if return_singular_values:
        data = (data, pca_op.singular_values_)
    return data