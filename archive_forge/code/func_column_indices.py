import torch
from xformers.ops import masked_matmul
from xformers.sparse import SparseCSRTensor
from xformers.sparse.utils import _csr_to_coo, _dense_to_sparse  # noqa: F401
@property
def column_indices(self):
    return self._mat._csr_column_indices