from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
def partitioned_size(p: Parameter) -> int:
    return p.partitioned_size() if RequirementCache('deepspeed<0.6.6') else p.partition_numel()