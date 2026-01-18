from scipy._lib.deprecation import _deprecated
from scipy._lib._finite_differences import _central_diff_weights, _derivative
from numpy import array, frombuffer, load
@_deprecated(msg='scipy.misc.ascent has been deprecated in SciPy v1.10.0; and will be completely removed in SciPy v1.12.0. Dataset methods have moved into the scipy.datasets module. Use scipy.datasets.ascent instead.')
def ascent():
    """
    Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy use in demos

    The image is derived from accent-to-the-top.jpg at
    http://www.public-domain-image.com/people-public-domain-images-pictures/

    .. deprecated:: 1.10.0
        `ascent` has been deprecated from `scipy.misc.ascent`
        in SciPy 1.10.0 and it will be completely removed in SciPy 1.12.0.
        Dataset methods have moved into the `scipy.datasets` module.
        Use `scipy.datasets.ascent` instead.

    Parameters
    ----------
    None

    Returns
    -------
    ascent : ndarray
       convenient image to use for testing and demonstration

    Examples
    --------
    >>> import scipy.misc
    >>> ascent = scipy.misc.ascent()
    >>> ascent.shape
    (512, 512)
    >>> ascent.max()
    255

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(ascent)
    >>> plt.show()

    """
    import pickle
    import os
    fname = os.path.join(os.path.dirname(__file__), 'ascent.dat')
    with open(fname, 'rb') as f:
        ascent = array(pickle.load(f))
    return ascent