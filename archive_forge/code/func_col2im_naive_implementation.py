import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
def col2im_naive_implementation(data, image_shape, kernel_shape, dilations, pads, strides):
    """Naive implementation for `col2im`."""
    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    _col2im_shape_check(data, image_shape, kernel_shape, dilations, new_pads, strides)
    data_col = data
    data_im = np.zeros(image_shape, dtype=data.dtype)
    dim_col = []
    for i in range(n_dims):
        col = (image_shape[i] + new_pads[i, :].sum() - (dilations[i] * (kernel_shape[i] - 1) + 1)) // strides[i] + 1
        dim_col.append(col)
    kernel_size = np.prod(kernel_shape)
    col_size = np.prod(dim_col)
    for c_col in range(kernel_size):
        offset = _get_indices(c_col, kernel_shape)
        for col in range(col_size):
            ind_col = _get_indices(col, dim_col)
            ind_im = []
            for i in range(n_dims):
                ind = ind_col[i] * strides[i] - new_pads[i, 0] + offset[i] * dilations[i]
                ind_im.append(ind)
            if not _is_out(ind_im, data_im.shape):
                data_im[tuple(ind_im)] += data_col[c_col, col]
    return data_im