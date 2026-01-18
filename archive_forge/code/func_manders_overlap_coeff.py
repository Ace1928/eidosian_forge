import numpy as np
from scipy.stats import pearsonr
from .._shared.utils import check_shape_equality, as_binary_ndarray
def manders_overlap_coeff(image0, image1, mask=None):
    """Manders' overlap coefficient

    Parameters
    ----------
    image0 : (M, N) ndarray
        Image of channel A. All pixel values should be non-negative.
    image1 : (M, N) ndarray
        Image of channel B. All pixel values should be non-negative.
        Must have same dimensions as `image0`
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0` and `image1` pixel values within this region of interest
        mask are included in the calculation.
        Must have ♣same dimensions as `image0`.

    Returns
    -------
    moc: float
        Manders' Overlap Coefficient of pixel intensities between the two
        images.

    Notes
    -----
    Manders' Overlap Coefficient (MOC) is given by the equation [1]_:

    .. math::
        r = \\frac{\\sum A_i B_i}{\\sqrt{\\sum A_i^2 \\sum B_i^2}}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`
        :math:`B_i` is the value of the :math:`i^{th}` pixel in `image1`

    It ranges between 0 for no colocalization and 1 for complete colocalization
    of all pixels.

    MOC does not take into account pixel intensities, just the fraction of
    pixels that have positive values for both channels[2]_ [3]_. Its usefulness
    has been criticized as it changes in response to differences in both
    co-occurence and correlation and so a particular MOC value could indicate
    a wide range of colocalization patterns [4]_ [5]_.

    References
    ----------
    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of
           co-localization of objects in dual-colour confocal images. Journal
           of Microscopy, 169: 375-382.
           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x
           https://imagej.net/media/manders.pdf
    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical
           guide to evaluating colocalization in biological microscopy.
           American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010
    .. [3] Bolte, S. and Cordelières, F.P. (2006), A guided tour into
           subcellular colocalization analysis in light microscopy. Journal of
           Microscopy, 224: 213-232.
           https://doi.org/10.1111/j.1365-2818.2006.01
    .. [4] Adler J, Parmryd I. (2010), Quantifying colocalization by
           correlation: the Pearson correlation coefficient is
           superior to the Mander's overlap coefficient. Cytometry A.
           Aug;77(8):733-42.https://doi.org/10.1002/cyto.a.20896
    .. [5] Adler, J, Parmryd, I. Quantifying colocalization: The case for
           discarding the Manders overlap coefficient. Cytometry. 2021; 99:
           910– 920. https://doi.org/10.1002/cyto.a.24336

    """
    image0 = np.asarray(image0)
    image1 = np.asarray(image1)
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0, image1, mask)
        image0 = image0[mask]
        image1 = image1[mask]
    else:
        check_shape_equality(image0, image1)
    if image0.min() < 0:
        raise ValueError('image0 contains negative values')
    if image1.min() < 0:
        raise ValueError('image1 contains negative values')
    denom = (np.sum(np.square(image0)) * np.sum(np.square(image1))) ** 0.5
    return np.sum(np.multiply(image0, image1)) / denom