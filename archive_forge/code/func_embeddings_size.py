from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig
@property
def embeddings_size(self) -> int:
    if self.projection_dim > 0:
        return self.encode_proj.out_features
    return self.bert_model.config.hidden_size