import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def set_record_attn(self, record_attn):
    """
        Makes forward prop dump self-attention softmaxes to self.saved_attn_weights.

        Args:
            record_attn (`Union[bool,set]`):
                Either a set of layer indices indicating which layers to store, or a boolean value indicating Whether
                to dump all.
        """

    def _should_record_attn(layer_idx):
        if isinstance(record_attn, bool):
            return record_attn
        return layer_idx in record_attn
    for i, layer in enumerate(self._attn_mods):
        layer.attn.record_attn = _should_record_attn(i)
    if not record_attn:
        self.saved_attn_weights = []