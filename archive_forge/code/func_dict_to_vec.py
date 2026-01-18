import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul
def dict_to_vec(val_dict, var_offsets, var_sizes, vec_len):
    """Converts a map of variable id to value to a vector.

    Parameters
    ----------
    val_dict : dict
        A map of variable id to value.
    var_offsets : dict
        A map of variable id to offset in the vector.
    var_sizes : dict
        A map of variable id to variable size.
    vector : NumPy matrix
        The vector to store the values in.
    """
    vector = np.zeros(vec_len)
    for id_, value in val_dict.items():
        size = var_sizes[id_]
        offset = var_offsets[id_]
        for col in range(size[1]):
            if np.isscalar(value):
                vector[offset:size[0] + offset] = value
            else:
                vector[offset:size[0] + offset] = np.squeeze(value[:, col])
            offset += size[0]
    return vector