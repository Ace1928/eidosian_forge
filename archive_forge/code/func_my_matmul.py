import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def my_matmul(outputs: List[torch.Tensor], dst_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
    for w, scale_weight, o in zip(weights, scales_weights, outputs):
        with torch.cuda.stream(stream_factory()):
            if _is_fp8_dtype(w.dtype):
                output_amax = torch.empty(1, dtype=torch.float32, device=o.device)
                torch._scaled_mm(gathered_input[dst_rank], w.t(), out_dtype=o.dtype, scale_a=scale_gathered_input, scale_b=scale_weight, out=(o, output_amax))
            else:
                torch.matmul(gathered_input[dst_rank], w.t(), out=o)