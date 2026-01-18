import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_marian import MarianConfig
def set_decoder_input_embeddings(self, value):
    if self.config.share_encoder_decoder_embeddings:
        raise ValueError('`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings are shared with the encoder. In order to set the decoder input embeddings, you should simply set the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings.')
    self.decoder.embed_tokens = value