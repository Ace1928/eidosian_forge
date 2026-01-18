import torch
from xformers.ops import masked_matmul
from xformers.sparse import SparseCSRTensor
from xformers.sparse.utils import _csr_to_coo, _dense_to_sparse  # noqa: F401
def matmul_with_mask(self, a, b):
    return type(self)._wrap(masked_matmul(a, b, self._mat))