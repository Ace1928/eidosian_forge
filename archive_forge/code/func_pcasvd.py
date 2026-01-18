import numpy as np
def pcasvd(data, keepdim=0, demean=True):
    """principal components with svd

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : int
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    demean : bool
        if true, then the column mean is subtracted from the data

    Returns
    -------
    xreduced : ndarray, 2d, (nobs, nvars)
        projection of the data x on the kept eigenvectors
    factors : ndarray, 2d, (nobs, nfactors)
        factor matrix, given by np.dot(x, evecs)
    evals : ndarray, 2d, (nobs, nfactors)
        eigenvalues
    evecs : ndarray, 2d, (nobs, nfactors)
        eigenvectors, normalized if normalize is true

    See Also
    --------
    pca : principal component analysis using eigenvector decomposition

    Notes
    -----
    This does not have yet the normalize option of pca.

    """
    nobs, nvars = data.shape
    x = np.array(data)
    if demean:
        m = x.mean(0)
    else:
        m = 0
    x -= m
    U, s, v = np.linalg.svd(x.T, full_matrices=1)
    factors = np.dot(U.T, x.T).T
    if keepdim:
        xreduced = np.dot(factors[:, :keepdim], U[:, :keepdim].T) + m
    else:
        xreduced = data
        keepdim = nvars
        ('print reassigning keepdim to max', keepdim)
    evals = s ** 2 / (x.shape[0] - 1)
    return (xreduced, factors[:, :keepdim], evals[:keepdim], U[:, :keepdim])