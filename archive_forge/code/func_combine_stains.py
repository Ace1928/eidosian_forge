from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def combine_stains(stains, conv_matrix, *, channel_axis=-1):
    """Stain to RGB color space conversion.

    Parameters
    ----------
    stains : (..., C=3, ...) array_like
        The image in stain color space. By default, the final dimension denotes
        channels.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `stains` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Stain combination matrices available in the ``color`` module and their
    respective colorspace:

    * ``rgb_from_hed``: Hematoxylin + Eosin + DAB
    * ``rgb_from_hdx``: Hematoxylin + DAB
    * ``rgb_from_fgx``: Feulgen + Light Green
    * ``rgb_from_bex``: Giemsa stain : Methyl Blue + Eosin
    * ``rgb_from_rbd``: FastRed + FastBlue +  DAB
    * ``rgb_from_gdx``: Methyl Green + DAB
    * ``rgb_from_hax``: Hematoxylin + AEC
    * ``rgb_from_bro``: Blue matrix Anilline Blue + Red matrix Azocarmine                        + Orange matrix Orange-G
    * ``rgb_from_bpx``: Methyl Blue + Ponceau Fuchsin
    * ``rgb_from_ahx``: Alcian Blue + Hematoxylin
    * ``rgb_from_hpx``: Hematoxylin + PAS

    References
    ----------
    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html
    .. [2] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical
           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.
           23, no. 4, pp. 291–299, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import (separate_stains, combine_stains,
    ...                            hdx_from_rgb, rgb_from_hdx)
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    >>> ihc_rgb = combine_stains(ihc_hdx, rgb_from_hdx)
    """
    stains = _prepare_colorarray(stains, channel_axis=-1)
    log_adjust = -np.log(1e-06)
    log_rgb = -(stains * log_adjust) @ conv_matrix
    rgb = np.exp(log_rgb)
    return np.clip(rgb, a_min=0, a_max=1)