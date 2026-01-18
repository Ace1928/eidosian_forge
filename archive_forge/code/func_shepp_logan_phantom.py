import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def shepp_logan_phantom():
    """Shepp Logan Phantom.

    References
    ----------
    .. [1] L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head
           section," in IEEE Transactions on Nuclear Science, vol. 21,
           no. 3, pp. 21-43, June 1974. :DOI:`10.1109/TNS.1974.6499235`

    Returns
    -------
    phantom : (400, 400) float64 image
        Image of the Shepp-Logan phantom in grayscale.
    """
    return _load('data/phantom.png', as_gray=True)