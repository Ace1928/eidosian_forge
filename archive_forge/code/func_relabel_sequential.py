import numpy as np
from ..util._map_array import map_array, ArrayMap
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : ArrayMap
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The output data type will be the same as `relabeled`.
    inverse_map : ArrayMap
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The output data type will be the same as `label_field`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> print(fw)
    ArrayMap:
      1 → 1
      5 → 2
      8 → 3
      42 → 4
      99 → 5
    >>> np.array(fw)
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> np.array(inv)
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    if offset <= 0:
        raise ValueError('Offset must be strictly positive.')
    if np.min(label_field) < 0:
        raise ValueError('Cannot relabel array that contains negative values.')
    offset = int(offset)
    in_vals = np.unique(label_field)
    if in_vals[0] == 0:
        out_vals = np.concatenate([[0], np.arange(offset, offset + len(in_vals) - 1)])
    else:
        out_vals = np.arange(offset, offset + len(in_vals))
    input_type = label_field.dtype
    if input_type.kind not in 'iu':
        raise TypeError('label_field must have an integer dtype')
    required_type = np.min_scalar_type(out_vals[-1])
    if input_type.itemsize < required_type.itemsize:
        output_type = required_type
    elif out_vals[-1] < np.iinfo(input_type).max:
        output_type = input_type
    else:
        output_type = required_type
    out_array = np.empty(label_field.shape, dtype=output_type)
    out_vals = out_vals.astype(output_type)
    map_array(label_field, in_vals, out_vals, out=out_array)
    fw_map = ArrayMap(in_vals, out_vals)
    inv_map = ArrayMap(out_vals, in_vals)
    return (out_array, fw_map, inv_map)