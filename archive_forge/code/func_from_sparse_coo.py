import torch
from xformers.ops import masked_matmul
from xformers.sparse import _csr_ops
from xformers.sparse.utils import (
@classmethod
def from_sparse_coo(cls, arg0):
    """
        assert arg0.is_sparse
        x = arg0.coalesce()
        rows, cols = x.indices().unbind(0)
        vals = x.values()
        _coo_to_csr()
        """
    pass