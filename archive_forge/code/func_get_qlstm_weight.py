import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
from torch.fx import GraphModule
from torch.fx.graph import Node
from .utils import (
from .ns_types import (
from typing import List, Optional, Dict, Callable
def get_qlstm_weight(mod: nn.Module) -> List[torch.Tensor]:
    res = []
    for weight_value in mod._all_weight_values:
        res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
        res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
    return res