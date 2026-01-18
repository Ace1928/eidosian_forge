from typing import Optional
import torch
def bgmv(y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor, indicies: torch.LongTensor, layer_idx: int, scale: float):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ w_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      w_t_all: Shape: `[None, L, H2, H1]`. All of the transposed weight
        matrices.
      indicies: Shape: `[B]`. Indices of the weight matrices.
      layer_idx: Layer index of the weight matrices.
      scale: Scaling factor.
    """
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    punica_kernels.dispatch_bgmv(y, x, w_t_all, indicies, layer_idx, scale)