import numpy as np
def fisher_vector(descriptors, gmm, *, improved=False, alpha=0.5):
    """Compute the Fisher vector given some descriptors/vectors,
    and an associated estimated GMM.

    Parameters
    ----------
    descriptors : np.ndarray, shape=(n_descriptors, descriptor_length)
        NumPy array of the descriptors for which the Fisher vector
        representation is to be computed.
    gmm : :class:`sklearn.mixture.GaussianMixture`
        An estimated GMM object, which contains the necessary parameters needed
        to compute the Fisher vector.
    improved : bool, default=False
        Flag denoting whether to compute improved Fisher vectors or not.
        Improved Fisher vectors are L2 and power normalized. Power
        normalization is simply f(z) = sign(z) pow(abs(z), alpha) for some
        0 <= alpha <= 1.
    alpha : float, default=0.5
        The parameter for the power normalization step. Ignored if
        improved=False.

    Returns
    -------
    fisher_vector : np.ndarray
        The computation Fisher vector, which is given by a concatenation of the
        gradients of a GMM with respect to its parameters (mixture weights,
        means, and covariance matrices). For D-dimensional input descriptors or
        vectors, and a K-mode GMM, the Fisher vector dimensionality will be
        2KD + K. Thus, its dimensionality is invariant to the number of
        descriptors/vectors.

    References
    ----------
    .. [1] Perronnin, F. and Dance, C. Fisher kernels on Visual Vocabularies
           for Image Categorization, IEEE Conference on Computer Vision and
           Pattern Recognition, 2007
    .. [2] Perronnin, F. and Sanchez, J. and Mensink T. Improving the Fisher
           Kernel for Large-Scale Image Classification, ECCV, 2010

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('sklearn')

    >>> from skimage.feature import fisher_vector, learn_gmm
    >>> sift_for_images = [np.random.random((10, 128)) for _ in range(10)]
    >>> num_modes = 16
    >>> # Estimate 16-mode GMM with these synthetic SIFT vectors
    >>> gmm = learn_gmm(sift_for_images, n_modes=num_modes)
    >>> test_image_descriptors = np.random.random((25, 128))
    >>> # Compute the Fisher vector
    >>> fv = fisher_vector(test_image_descriptors, gmm)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError('scikit-learn is not installed. Please ensure it is installed in order to use the Fisher vector functionality.')
    if not isinstance(descriptors, np.ndarray):
        raise DescriptorException('Please ensure descriptors is a NumPy array.')
    if not isinstance(gmm, GaussianMixture):
        raise FisherVectorException('Please ensure gmm is a sklearn.mixture.GaussianMixture object.')
    if improved and (not isinstance(alpha, float)):
        raise FisherVectorException('Please ensure that the alpha parameter is a float.')
    num_descriptors = len(descriptors)
    mixture_weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    posterior_probabilities = gmm.predict_proba(descriptors)
    pp_sum = posterior_probabilities.mean(axis=0, keepdims=True).T
    pp_x = posterior_probabilities.T.dot(descriptors) / num_descriptors
    pp_x_2 = posterior_probabilities.T.dot(np.power(descriptors, 2)) / num_descriptors
    d_pi = pp_sum.squeeze() - mixture_weights
    d_mu = pp_x - pp_sum * means
    d_sigma_t1 = pp_sum * np.power(means, 2)
    d_sigma_t2 = pp_sum * covariances
    d_sigma_t3 = 2 * pp_x * means
    d_sigma = -pp_x_2 - d_sigma_t1 + d_sigma_t2 + d_sigma_t3
    sqrt_mixture_weights = np.sqrt(mixture_weights)
    d_pi /= sqrt_mixture_weights
    d_mu /= sqrt_mixture_weights[:, np.newaxis] * np.sqrt(covariances)
    d_sigma /= np.sqrt(2) * sqrt_mixture_weights[:, np.newaxis] * covariances
    fisher_vector = np.hstack((d_pi, d_mu.ravel(), d_sigma.ravel()))
    if improved:
        fisher_vector = np.sign(fisher_vector) * np.power(np.abs(fisher_vector), alpha)
        fisher_vector = fisher_vector / np.linalg.norm(fisher_vector)
    return fisher_vector