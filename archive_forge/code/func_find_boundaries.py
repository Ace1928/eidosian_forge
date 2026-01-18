import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..morphology import dilation, erosion, square
from ..util import img_as_float, view_as_windows
from ..color import gray2rgb
def find_boundaries(label_img, connectivity=1, mode='thick', background=0):
    """Return bool array where boundaries between labeled regions are True.

    Parameters
    ----------
    label_img : array of int or bool
        An array in which different regions are labeled with either different
        integers or boolean values.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:

        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.
    background : int, optional
        For modes 'inner' and 'outer', a definition of a background
        label is required. See `mode` for descriptions of these two.

    Returns
    -------
    boundaries : array of bool, same shape as `label_img`
        A bool image where ``True`` represents a boundary pixel. For
        `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).

    Examples
    --------
    >>> labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> find_boundaries(labels, mode='thick').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='inner').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='outer').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> labels_small = labels[::2, ::3]
    >>> labels_small
    array([[0, 0, 0, 0],
           [0, 0, 5, 0],
           [0, 1, 5, 0],
           [0, 0, 5, 0],
           [0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels_small, mode='subpixel').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> bool_image = np.array([[False, False, False, False, False],
    ...                        [False, False, False, False, False],
    ...                        [False, False,  True,  True,  True],
    ...                        [False, False,  True,  True,  True],
    ...                        [False, False,  True,  True,  True]],
    ...                       dtype=bool)
    >>> find_boundaries(bool_image)
    array([[False, False, False, False, False],
           [False, False,  True,  True,  True],
           [False,  True,  True,  True,  True],
           [False,  True,  True, False, False],
           [False,  True,  True, False, False]])
    """
    if label_img.dtype == 'bool':
        label_img = label_img.astype(np.uint8)
    ndim = label_img.ndim
    footprint = ndi.generate_binary_structure(ndim, connectivity)
    if mode != 'subpixel':
        boundaries = dilation(label_img, footprint) != erosion(label_img, footprint)
        if mode == 'inner':
            foreground_image = label_img != background
            boundaries &= foreground_image
        elif mode == 'outer':
            max_label = np.iinfo(label_img.dtype).max
            background_image = label_img == background
            footprint = ndi.generate_binary_structure(ndim, ndim)
            inverted_background = np.array(label_img, copy=True)
            inverted_background[background_image] = max_label
            adjacent_objects = (dilation(label_img, footprint) != erosion(inverted_background, footprint)) & ~background_image
            boundaries &= background_image | adjacent_objects
        return boundaries
    else:
        boundaries = _find_boundaries_subpixel(label_img)
        return boundaries