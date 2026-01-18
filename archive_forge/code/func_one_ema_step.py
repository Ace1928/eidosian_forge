import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def one_ema_step(self, inputs, past_state=None):
    damping_factor, previous_timestep_weight = self.get_ema_coefficients()
    updated_state = (damping_factor * self.ema_expansion_matrix).squeeze(-1) * inputs
    if past_state is not None:
        updated_state = updated_state + previous_timestep_weight.squeeze(-1) * past_state
    out = torch.einsum('bdn,dn->bd', updated_state, self.kernel_projection_matrix * self.scale)
    return (out.unsqueeze(0), updated_state)