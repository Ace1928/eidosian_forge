from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def dense_to_sst(self, dense: Tensor) -> Optional[Tensor]:
    """Get Signal Sparse Tensor (SST) from a dense tensor

        Dense -> fft -> top-k -> results.

        The input dense tensor is transformed using a transform algorithm according to the `algo`
        initialization argument. The SST is then generated from the top_k_elements
        (or the top_k_percentage) of values from the transformed tensor along the 'sst_top_k_dim'.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).

        Returns:
            (Tensor, optional):
                Same shaped tensor as the input dense tensor, still in dense format but in frequency
                domain (complex valued) and has zeros.
        """
    if not self._sst_enabled:
        return None
    top_k_total_size = _top_k_total_size(dense, self._sst_top_k_dim)
    k = _get_k_for_topk(self._sst_top_k_percent, self._sst_top_k_element, top_k_total_size)
    dense_freq = self._transform(dense, dim=self._sst_top_k_dim)
    real_dense_freq = dense_freq.real.abs()
    return _scatter_topk_to_sparse_tensor(real_dense_freq, dense_freq, k, dim=self._sst_top_k_dim)