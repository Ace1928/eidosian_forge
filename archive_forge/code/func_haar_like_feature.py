from itertools import chain
from operator import add
import numpy as np
from ._haar import haar_like_feature_coord_wrapper
from ._haar import haar_like_feature_wrapper
from ..color import gray2rgb
from ..draw import rectangle
from ..util import img_as_float
def haar_like_feature(int_image, r, c, width, height, feature_type=None, feature_coord=None):
    """Compute the Haar-like features for a region of interest (ROI) of an
    integral image.

    Haar-like features have been successfully used for image classification and
    object detection [1]_. It has been used for real-time face detection
    algorithm proposed in [2]_.

    Parameters
    ----------
    int_image : (M, N) ndarray
        Integral image for which the features need to be computed.
    r : int
        Row-coordinate of top left corner of the detection window.
    c : int
        Column-coordinate of top left corner of the detection window.
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_type : str or list of str or None, optional
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

        By default all features are extracted.

        If using with `feature_coord`, it should correspond to the feature
        type of each associated coordinate feature.
    feature_coord : ndarray of list of tuples or None, optional
        The array of coordinates to be extracted. This is useful when you want
        to recompute only a subset of features. In this case `feature_type`
        needs to be an array containing the type of each feature, as returned
        by :func:`haar_like_feature_coord`. By default, all coordinates are
        computed.

    Returns
    -------
    haar_features : (n_features,) ndarray of int or float
        Resulting Haar-like features. Each value is equal to the subtraction of
        sums of the positive and negative rectangles. The data type depends of
        the data type of `int_image`: `int` when the data type of `int_image`
        is `uint` or `int` and `float` when the data type of `int_image` is
        `float`.

    Notes
    -----
    When extracting those features in parallel, be aware that the choice of the
    backend (i.e. multiprocessing vs threading) will have an impact on the
    performance. The rule of thumb is as follows: use multiprocessing when
    extracting features for all possible ROI in an image; use threading when
    extracting the feature at specific location for a limited number of ROIs.
    Refer to the example
    :ref:`sphx_glr_auto_examples_applications_plot_haar_extraction_selection_classification.py`
    for more insights.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import haar_like_feature
    >>> img = np.ones((5, 5), dtype=np.uint8)
    >>> img_ii = integral_image(img)
    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-3-x')
    >>> feature
    array([-1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2,
           -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -1,
           -2, -3, -1, -2, -1, -2, -1, -2, -1, -1, -1])

    You can compute the feature for some pre-computed coordinates.

    >>> from skimage.feature import haar_like_feature_coord
    >>> feature_coord, feature_type = zip(
    ...     *[haar_like_feature_coord(5, 5, feat_t)
    ...       for feat_t in ('type-2-x', 'type-3-x')])
    >>> # only select one feature over two
    >>> feature_coord = np.concatenate([x[::2] for x in feature_coord])
    >>> feature_type = np.concatenate([x[::2] for x in feature_type])
    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5,
    ...                             feature_type=feature_type,
    ...                             feature_coord=feature_coord)
    >>> feature
    array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -3, -5, -2, -4, -1,
           -3, -5, -2, -4, -2, -4, -2, -4, -2, -1, -3, -2, -1, -1, -1, -1, -1])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Haar-like_feature
    .. [2] Oren, M., Papageorgiou, C., Sinha, P., Osuna, E., & Poggio, T.
           (1997, June). Pedestrian detection using wavelet templates.
           In Computer Vision and Pattern Recognition, 1997. Proceedings.,
           1997 IEEE Computer Society Conference on (pp. 193-199). IEEE.
           http://tinyurl.com/y6ulxfta
           :DOI:`10.1109/CVPR.1997.609319`
    .. [3] Viola, Paul, and Michael J. Jones. "Robust real-time face
           detection." International journal of computer vision 57.2
           (2004): 137-154.
           https://www.merl.com/publications/docs/TR2004-043.pdf
           :DOI:`10.1109/CVPR.2001.990517`

    """
    if feature_coord is None:
        feature_type_ = _validate_feature_type(feature_type)
        return np.hstack(list(chain.from_iterable((haar_like_feature_wrapper(int_image, r, c, width, height, feat_t, feature_coord) for feat_t in feature_type_))))
    else:
        if feature_coord.shape[0] != feature_type.shape[0]:
            raise ValueError('Inconsistent size between feature coordinatesand feature types.')
        mask_feature = [feature_type == feat_t for feat_t in FEATURE_TYPE]
        haar_feature_idx, haar_feature = zip(*[(np.flatnonzero(mask), haar_like_feature_wrapper(int_image, r, c, width, height, feat_t, feature_coord[mask])) for mask, feat_t in zip(mask_feature, FEATURE_TYPE) if np.count_nonzero(mask)])
        haar_feature_idx = np.concatenate(haar_feature_idx)
        haar_feature = np.concatenate(haar_feature)
        haar_feature[haar_feature_idx] = haar_feature.copy()
        return haar_feature