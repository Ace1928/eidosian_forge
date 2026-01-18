import numpy as np
from ..util._map_array import map_array, ArrayMap
def join_segmentations(s1, s2, return_mapping: bool=False):
    """Return the join of the two input segmentations.

    The join J of S1 and S2 is defined as the segmentation in which two
    voxels are in the same segment if and only if they are in the same
    segment in *both* S1 and S2.

    Parameters
    ----------
    s1, s2 : numpy arrays
        s1 and s2 are label fields of the same shape.
    return_mapping : bool, optional
        If true, return mappings for joined segmentation labels to the original labels.

    Returns
    -------
    j : numpy array
        The join segmentation of s1 and s2.
    map_j_to_s1 : ArrayMap, optional
        Mapping from labels of the joined segmentation j to labels of s1.
    map_j_to_s2 : ArrayMap, optional
        Mapping from labels of the joined segmentation j to labels of s2.

    Examples
    --------
    >>> from skimage.segmentation import join_segmentations
    >>> s1 = np.array([[0, 0, 1, 1],
    ...                [0, 2, 1, 1],
    ...                [2, 2, 2, 1]])
    >>> s2 = np.array([[0, 1, 1, 0],
    ...                [0, 1, 1, 0],
    ...                [0, 1, 1, 1]])
    >>> join_segmentations(s1, s2)
    array([[0, 1, 3, 2],
           [0, 5, 3, 2],
           [4, 5, 5, 3]])
    >>> j, m1, m2 = join_segmentations(s1, s2, return_mapping=True)
    >>> m1
    ArrayMap(array([0, 1, 2, 3, 4, 5]), array([0, 0, 1, 1, 2, 2]))
    >>> np.all(m1[j] == s1)
    True
    >>> np.all(m2[j] == s2)
    True
    """
    if s1.shape != s2.shape:
        raise ValueError(f'Cannot join segmentations of different shape. s1.shape: {s1.shape}, s2.shape: {s2.shape}')
    s1_relabeled, _, backward_map1 = relabel_sequential(s1)
    s2_relabeled, _, backward_map2 = relabel_sequential(s2)
    factor = s2.max() + 1
    j_initial = factor * s1_relabeled + s2_relabeled
    j, _, map_j_to_j_initial = relabel_sequential(j_initial)
    if not return_mapping:
        return j
    labels_j = np.unique(j_initial)
    labels_s1_relabeled, labels_s2_relabeled = np.divmod(labels_j, factor)
    map_j_to_s1 = ArrayMap(map_j_to_j_initial.in_values, backward_map1[labels_s1_relabeled])
    map_j_to_s2 = ArrayMap(map_j_to_j_initial.in_values, backward_map2[labels_s2_relabeled])
    return (j, map_j_to_s1, map_j_to_s2)