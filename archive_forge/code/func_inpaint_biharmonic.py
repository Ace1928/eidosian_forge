import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner
@utils.channel_as_last_axis()
def inpaint_biharmonic(image, mask, *, split_into_regions=False, channel_axis=None):
    """Inpaint masked points in image with biharmonic equations.

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    mask : (M[, N[, ..., P]]) ndarray
        Array of pixels to be inpainted. Have to be the same shape as one
        of the 'image' channels. Unknown pixels have to be represented with 1,
        known pixels - with 0.
    split_into_regions : boolean, optional
        If True, inpainting is performed on a region-by-region basis. This is
        likely to be slower, but will have reduced memory requirements.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (M[, N[, ..., P]][, C]) ndarray
        Input image with masked pixels inpainted.

    References
    ----------
    .. [1]  S.B.Damelin and N.S.Hoang. "On Surface Completion and Image
            Inpainting by Biharmonic Functions: Numerical Aspects",
            International Journal of Mathematics and Mathematical Sciences,
            Vol. 2018, Article ID 3950312
            :DOI:`10.1155/2018/3950312`
    .. [2]  C. K. Chui and H. N. Mhaskar, MRA Contextual-Recovery Extension of
            Smooth Functions on Manifolds, Appl. and Comp. Harmonic Anal.,
            28 (2010), 104-113,
            :DOI:`10.1016/j.acha.2009.04.004`

    Examples
    --------
    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    >>> mask = np.zeros_like(img)
    >>> mask[2, 2:] = 1
    >>> mask[1, 3:] = 1
    >>> mask[0, 4:] = 1
    >>> out = inpaint_biharmonic(img, mask)
    """
    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')
    multichannel = channel_axis is not None
    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')
    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')
    image = skimage.img_as_float(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    mask = mask.astype(bool, copy=False)
    if not multichannel:
        image = image[..., np.newaxis]
    out = np.copy(image, order='C')
    radius = 2
    coef_shape = (2 * radius + 1,) * mask.ndim
    coef_center = (radius,) * mask.ndim
    neigh_coef_full, coef_idx, coef_vals = _get_neigh_coef(coef_shape, coef_center, dtype=out.dtype)
    channel_stride_bytes = out.strides[-2]
    offsets = coef_idx - radius
    known_points = image[~mask]
    limits = (known_points.min(axis=0), known_points.max(axis=0))
    if split_into_regions:
        kernel = ndi.generate_binary_structure(mask.ndim, 1)
        mask_dilated = ndi.binary_dilation(mask, structure=kernel)
        mask_labeled = label(mask_dilated)
        mask_labeled *= mask
        bbox_slices = ndi.find_objects(mask_labeled)
        for idx_region, bb_slice in enumerate(bbox_slices, 1):
            roi_sl = tuple((slice(max(sl.start - radius, 0), min(sl.stop + radius, size)) for sl, size in zip(bb_slice, mask_labeled.shape)))
            mask_region = mask_labeled[roi_sl] == idx_region
            roi_sl += (slice(None),)
            otmp = out[roi_sl].copy()
            ostrides = np.array([s // channel_stride_bytes for s in otmp[..., 0].strides])
            raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)
            _inpaint_biharmonic_single_region(image[roi_sl], mask_region, otmp, neigh_coef_full, coef_vals, raveled_offsets)
            out[roi_sl] = otmp
    else:
        ostrides = np.array([s // channel_stride_bytes for s in out[..., 0].strides])
        raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)
        _inpaint_biharmonic_single_region(image, mask, out, neigh_coef_full, coef_vals, raveled_offsets)
    np.clip(out, a_min=limits[0], a_max=limits[1], out=out)
    if not multichannel:
        out = out[..., 0]
    return out