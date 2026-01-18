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
def bert_mask(self, aa, esmaa, mask, pattern):
    new_aa = aa.clone()
    target = aa.clone()
    new_esmaa = esmaa.clone()
    new_aa[pattern == 1] = self.mask_idx
    target[pattern != 1] = 0
    new_esmaa[pattern == 1] = self.esm_dict_mask_idx
    return (new_aa, new_esmaa, target)