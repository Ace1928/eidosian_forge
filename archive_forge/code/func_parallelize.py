import warnings
from typing import Optional, Tuple, Union
import torch
import torch.fx
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gptj import GPTJConfig
@add_start_docstrings(PARALLELIZE_DOCSTRING)
def parallelize(self, device_map=None):
    warnings.warn("`GPTJForCausalLM.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0': 0, 'transformer.h.1': 1, ...}", FutureWarning)
    self.device_map = get_device_map(len(self.transformer.h), range(torch.cuda.device_count())) if device_map is None else device_map
    assert_device_map(self.device_map, len(self.transformer.h))
    self.transformer.parallelize(self.device_map)
    self.lm_head = self.lm_head.to(self.transformer.first_device)
    self.model_parallel = True