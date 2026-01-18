import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
def compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
    device = next(self.parameters()).device
    B, L = esmaa.shape
    if self.config.esmfold_config.bypass_lm:
        esm_s = torch.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats, device=device)
        return esm_s
    bosi, eosi = (self.esm_dict_cls_idx, self.esm_dict_eos_idx)
    bos = esmaa.new_full((B, 1), bosi)
    eos = esmaa.new_full((B, 1), self.esm_dict_padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    esmaa[range(B), (esmaa != 1).sum(1)] = eosi
    esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)['hidden_states']
    esm_s = torch.stack(esm_hidden_states, dim=2)
    esm_s = esm_s[:, 1:-1]
    return esm_s