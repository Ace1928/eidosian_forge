from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def dense_sst_to_dst(self, dense: Tensor, sst: Optional[Tensor]) -> Optional[Tensor]:
    """Calculates DST from input dense and SST tensors.

        dense - inverse_transform(sst)[using sst_dst_to_dense method] -> top-k -> dst

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).
            sst (Tensor):
                Input SST tensor (has zeros).

        Returns:
            (Tensor):
                Same shaped tensor, still dense format but has zeros. Non-zeros are top-k delta values.
        """
    if not self._dst_enabled:
        return None
    if sst is None:
        sst = torch.zeros_like(dense, dtype=torch.complex64)
    if not dense.shape == sst.shape:
        raise ValueError('dense and sst have different shapes!')
    top_k_total_size = _top_k_total_size(dense, self._dst_top_k_dim)
    k = _get_k_for_topk(self._dst_top_k_percent, self._dst_top_k_element, top_k_total_size)
    delta = dense - self.sst_dst_to_dense(sst)
    del dense
    return _scatter_topk_to_sparse_tensor(delta.abs(), delta, k, dim=self._dst_top_k_dim)