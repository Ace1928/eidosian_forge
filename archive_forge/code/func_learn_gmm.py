import numpy as np
def learn_gmm(descriptors, *, n_modes=32, gm_args=None):
    """Estimate a Gaussian mixture model (GMM) given a set of descriptors and
    number of modes (i.e. Gaussians). This function is essentially a wrapper
    around the scikit-learn implementation of GMM, namely the
    :class:`sklearn.mixture.GaussianMixture` class.

    Due to the nature of the Fisher vector, the only enforced parameter of the
    underlying scikit-learn class is the covariance_type, which must be 'diag'.

    There is no simple way to know what value to use for `n_modes` a-priori.
    Typically, the value is usually one of ``{16, 32, 64, 128}``. One may train
    a few GMMs and choose the one that maximises the log probability of the
    GMM, or choose `n_modes` such that the downstream classifier trained on
    the resultant Fisher vectors has maximal performance.

    Parameters
    ----------
    descriptors : np.ndarray (N, M) or list [(N1, M), (N2, M), ...]
        List of NumPy arrays, or a single NumPy array, of the descriptors
        used to estimate the GMM. The reason a list of NumPy arrays is
        permissible is because often when using a Fisher vector encoding,
        descriptors/vectors are computed separately for each sample/image in
        the dataset, such as SIFT vectors for each image. If a list if passed
        in, then each element must be a NumPy array in which the number of
        rows may differ (e.g. different number of SIFT vector for each image),
        but the number of columns for each must be the same (i.e. the
        dimensionality must be the same).
    n_modes : int
        The number of modes/Gaussians to estimate during the GMM estimate.
    gm_args : dict
        Keyword arguments that can be passed into the underlying scikit-learn
        :class:`sklearn.mixture.GaussianMixture` class.

    Returns
    -------
    gmm : :class:`sklearn.mixture.GaussianMixture`
        The estimated GMM object, which contains the necessary parameters
        needed to compute the Fisher vector.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('sklearn')

    >>> from skimage.feature import fisher_vector
    >>> rng = np.random.Generator(np.random.PCG64())
    >>> sift_for_images = [rng.standard_normal((10, 128)) for _ in range(10)]
    >>> num_modes = 16
    >>> # Estimate 16-mode GMM with these synthetic SIFT vectors
    >>> gmm = learn_gmm(sift_for_images, n_modes=num_modes)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError('scikit-learn is not installed. Please ensure it is installed in order to use the Fisher vector functionality.')
    if not isinstance(descriptors, list | np.ndarray):
        raise DescriptorException('Please ensure descriptors are either a NumPY array, or a list of NumPy arrays.')
    d_mat_1 = descriptors[0]
    if isinstance(descriptors, list) and (not isinstance(d_mat_1, np.ndarray)):
        raise DescriptorException('Please ensure descriptors are a list of NumPy arrays.')
    if isinstance(descriptors, list):
        expected_shape = descriptors[0].shape
        ranks = [len(e.shape) == len(expected_shape) for e in descriptors]
        if not all(ranks):
            raise DescriptorException('Please ensure all elements of your descriptor list are of rank 2.')
        dims = [e.shape[1] == descriptors[0].shape[1] for e in descriptors]
        if not all(dims):
            raise DescriptorException('Please ensure all descriptors are of the same dimensionality.')
    if not isinstance(n_modes, int) or n_modes <= 0:
        raise FisherVectorException('Please ensure n_modes is a positive integer.')
    if gm_args:
        has_cov_type = 'covariance_type' in gm_args
        cov_type_not_diag = gm_args['covariance_type'] != 'diag'
        if has_cov_type and cov_type_not_diag:
            raise FisherVectorException('Covariance type must be "diag".')
    if isinstance(descriptors, list):
        descriptors = np.vstack(descriptors)
    if gm_args:
        has_cov_type = 'covariance_type' in gm_args
        if has_cov_type:
            gmm = GaussianMixture(n_components=n_modes, **gm_args)
        else:
            gmm = GaussianMixture(n_components=n_modes, covariance_type='diag', **gm_args)
    else:
        gmm = GaussianMixture(n_components=n_modes, covariance_type='diag')
    gmm.fit(descriptors)
    return gmm