import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def scatter_nd_impl(data, indices, updates, reduction='none'):
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        if reduction == 'add':
            output[tuple(indices[i])] += updates[i]
        elif reduction == 'mul':
            output[tuple(indices[i])] *= updates[i]
        elif reduction == 'max':
            output[tuple(indices[i])] = np.maximum(output[indices[i]], updates[i])
        elif reduction == 'min':
            output[tuple(indices[i])] = np.minimum(output[indices[i]], updates[i])
        else:
            output[tuple(indices[i])] = updates[i]
    return output