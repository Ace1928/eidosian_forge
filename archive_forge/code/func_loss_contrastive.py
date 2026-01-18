import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def loss_contrastive(self, contrastive_queries_logits: Tensor, text_queries: Tensor):
    """Compute the query-text contrastive loss.

        Args:
            contrastive_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
            text_queries (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_contrastive** -- The query-text contrastive loss computed using task-guided queries
                                    and text queries derived from input text list.
        """
    image_queries = contrastive_queries_logits.float()
    image_queries = nn.functional.normalize(image_queries.flatten(1), dim=-1)
    text_queries = nn.functional.normalize(text_queries.flatten(1), dim=-1)
    logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
    logits_per_text = torch.matmul(text_queries, image_queries.t()) * logit_scale
    logits_per_img = logits_per_text.t()
    loss_img = nn.functional.cross_entropy(logits_per_img, torch.arange(len(logits_per_img), device=logits_per_text.device))
    loss_text = nn.functional.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device))
    loss_contrastive = loss_img + loss_text
    losses = {'loss_contrastive': loss_contrastive}
    return losses