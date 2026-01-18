import numpy as np
from .loadsave import load
from .orientations import OrientationError, io_orientation
def concat_images(images, check_affines=True, axis=None):
    """Concatenate images in list to single image, along specified dimension

    Parameters
    ----------
    images : sequence
       sequence of ``SpatialImage`` or filenames of the same dimensionality\\s
    check_affines : {True, False}, optional
       If True, then check that all the affines for `images` are nearly
       the same, raising a ``ValueError`` otherwise.  Default is True
    axis : None or int, optional
        If None, concatenates on a new dimension.  This requires all images to
        be the same shape.  If not None, concatenates on the specified
        dimension.  This requires all images to be the same shape, except on
        the specified dimension.

    Returns
    -------
    concat_img : ``SpatialImage``
       New image resulting from concatenating `images` across last
       dimension
    """
    images = [load(img) if not hasattr(img, 'get_data') else img for img in images]
    n_imgs = len(images)
    if n_imgs == 0:
        raise ValueError('Cannot concatenate an empty list of images.')
    img0 = images[0]
    affine = img0.affine
    header = img0.header
    klass = img0.__class__
    shape0 = img0.shape
    n_dim = len(shape0)
    if axis is None:
        out_shape = (n_imgs,) + shape0
        out_data = np.empty(out_shape)
    else:
        out_data = [None] * n_imgs
    idx_mask = np.ones((n_dim,), dtype=bool)
    if axis is not None:
        idx_mask[axis] = False
    masked_shape = np.array(shape0)[idx_mask]
    for i, img in enumerate(images):
        if len(img.shape) != n_dim:
            raise ValueError(f'Image {i} has {len(img.shape)} dimensions, image 0 has {n_dim}')
        if not np.all(np.array(img.shape)[idx_mask] == masked_shape):
            raise ValueError(f'shape {img.shape} for image {i} not compatible with first image shape {shape0} with axis == {axis}')
        if check_affines and (not np.all(img.affine == affine)):
            raise ValueError(f'Affine for image {i} does not match affine for first image')
        out_data[i] = np.asanyarray(img.dataobj)
    if axis is None:
        out_data = np.rollaxis(out_data, 0, out_data.ndim)
    else:
        out_data = np.concatenate(out_data, axis=axis)
    return klass(out_data, affine, header)