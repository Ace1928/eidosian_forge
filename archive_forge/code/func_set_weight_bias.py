from typing import Optional
import torch
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.sparse.quantized import linear
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern
from torch.ao.nn.quantized.modules.utils import _quantize_weight, _hide_packed_params_repr
def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor], row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
    assert row_block_size is not None and col_block_size is not None
    self.out_features = w.shape[0]
    self.in_features = w.shape[1]
    self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)