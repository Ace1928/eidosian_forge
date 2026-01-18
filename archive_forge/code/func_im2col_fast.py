import numpy as np
from onnx.reference.op_run import OpRun
def im2col_fast(X, kernel_shape, pads, strides):
    n_dims = len(kernel_shape)
    m, n_C = X.shape[:2]
    kernel_size = np.prod(kernel_shape)
    shape_out = []
    for i, dim in enumerate(kernel_shape):
        dx = X.shape[2 + i]
        shape_out.append((dx + pads[i] + pads[i + n_dims] - dim) // strides[i] + 1)
    indices = []
    for i in range(len(shape_out)):
        kind = _make_ind(i, kernel_shape)
        iind = _make_ind(i, shape_out) * strides[i]
        index = np.tile(kind.ravel(), n_C).reshape(-1, 1) + iind.reshape(1, -1)
        indices.append(index)
    d = np.repeat(np.arange(n_C), kernel_size).reshape(-1, 1)
    nc = [(0, 0)] * 2
    padding = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
    X_padded = np.pad(X, tuple(nc) + tuple(padding), mode='constant')
    getitem = (slice(0, m), d, *indices)
    cols = X_padded[getitem]
    conc_cols = np.concatenate(cols, axis=-1)
    return (conc_cols, tuple(shape_out))