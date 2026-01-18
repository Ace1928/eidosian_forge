import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
def class_predictor(self, image_feats: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor]=None, query_mask: Optional[torch.Tensor]=None) -> Tuple[torch.FloatTensor]:
    """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
    pred_logits, image_class_embeds = self.class_head(image_feats, query_embeds, query_mask)
    return (pred_logits, image_class_embeds)