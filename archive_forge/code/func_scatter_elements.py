import numpy as np
from onnx.reference.op_run import OpRun
def scatter_elements(data, indices, updates, axis=0, reduction=None):
    """Scatter elements.

    ::
        for 3-dim and axis=0
            output[indices[i][j][k]][j][k] = updates[i][j][k]
        for axis 1
            output[i][indices[i][j][k]][k] = updates[i][j][k]
        and so on.
    """
    if reduction == 'add':

        def f(x, y):
            return x + y
    elif reduction == 'min':

        def f(x, y):
            return min(x, y)
    elif reduction == 'max':

        def f(x, y):
            return max(x, y)
    else:

        def f(x, y):
            return y
    if axis < 0:
        axis = data.ndim + axis
    if len(data.shape) == 1 and axis == 0:
        scattered = np.copy(data)
        for pos, up in zip(indices, updates):
            scattered[pos] = f(scattered[pos], up)
        return scattered
    if len(indices.shape) == 2:
        scattered = np.copy(data)
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    scattered[indices[i, j], j] = f(scattered[indices[i, j], j], updates[i, j])
        else:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    scattered[i, indices[i, j]] = f(scattered[i, indices[i, j]], updates[i, j])
        return scattered
    if len(indices.shape) == 3:
        scattered = np.copy(data)
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[indices[i, j, k], j, k] = f(scattered[indices[i, j, k], j, k], updates[i, j, k])
        elif axis == 1:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[i, indices[i, j, k], k] = f(scattered[i, indices[i, j, k], k], updates[i, j, k])
        elif axis == 2:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[i, j, indices[i, j, k]] = f(scattered[i, j, indices[i, j, k]], updates[i, j, k])
        return scattered
    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1:]

    def make_slice(arr, axis, i):
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = (unpacked, packed[i])
        return unpacked
    idx = [[unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)), indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]] for i in range(indices.shape[axis])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape)))
    scattered = np.copy(data)
    if reduction == 'min':
        scattered[tuple(idx)] = np.minimum(scattered[tuple(idx)], updates[tuple(updates_idx)])
    elif reduction == 'max':
        scattered[tuple(idx)] = np.maximum(scattered[tuple(idx)], updates[tuple(updates_idx)])
    elif reduction == 'add':
        scattered[tuple(idx)] += updates[tuple(updates_idx)]
    else:
        scattered[tuple(idx)] = updates[tuple(updates_idx)]
    return scattered