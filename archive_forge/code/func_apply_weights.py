from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
def apply_weights(self, weights: Dict[str, Any], x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
    qweight = weights['B']
    scales = weights['s']
    workspace = weights['workspace']
    x_2d = x.view(-1, x.shape[-1])
    size_m = x_2d.shape[0]
    size_k = x_2d.shape[1]
    size_n = scales.shape[1]
    output_2d = ops.marlin_gemm(x_2d, qweight, scales, workspace, size_m, size_n, size_k)
    output = output_2d.view(x.shape[:-1] + (output_2d.shape[1],))
    if bias is not None:
        output.add_(bias)
    return output