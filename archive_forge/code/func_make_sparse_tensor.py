import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_sparse_tensor(values: TensorProto, indices: TensorProto, dims: Sequence[int]) -> SparseTensorProto:
    """Construct a SparseTensorProto

    Args:
        values (TensorProto): the values
        indices (TensorProto): the indices
        dims: the shape

    Returns:
        SparseTensorProto
    """
    sparse = SparseTensorProto()
    sparse.values.CopyFrom(values)
    sparse.indices.CopyFrom(indices)
    sparse.dims.extend(dims)
    return sparse