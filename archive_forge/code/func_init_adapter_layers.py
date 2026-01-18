import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
def init_adapter_layers(self):
    """
        (Re-)initialize attention adapter layers and lm head for adapter-only fine-tuning
        """
    for module in self.modules():
        if isinstance(module, Wav2Vec2AttnAdapterLayer):
            self._init_weights(module)
    if isinstance(self, Wav2Vec2ForCTC):
        self._init_weights(self.lm_head)