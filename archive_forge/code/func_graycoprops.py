import warnings
import numpy as np
from .._shared.utils import check_nD
from ..color import gray2rgb
from ..util import img_as_float
from ._texture import _glcm_loop, _local_binary_pattern, _multiblock_lbp
def graycoprops(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    Compute a feature of a gray level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]

    Each GLCM is normalized to have a sum of 1 before the computation of
    texture properties.

    .. versionchanged:: 0.19
           `greycoprops` was renamed to `graycoprops` in 0.19.

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy',             'correlation', 'ASM'}, optional
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = graycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])

    """
    check_nD(P, 4, 'P')
    num_level, num_level2, num_dist, num_angle = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1.0 / (1.0 + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    else:
        raise ValueError(f'{prop} is an invalid property')
    if prop == 'energy':
        asm = np.sum(P ** 2, axis=(0, 1))
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.sum(P ** 2, axis=(0, 1))
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * P, axis=(0, 1))
        diff_j = J - np.sum(J * P, axis=(0, 1))
        std_i = np.sqrt(np.sum(P * diff_i ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(P * diff_j ** 2, axis=(0, 1)))
        cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum(P * weights, axis=(0, 1))
    return results