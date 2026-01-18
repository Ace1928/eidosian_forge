import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
@torch.no_grad()
def loss_cardinality(self, outputs, targets, indices, num_boxes):
    """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
    logits = outputs['logits']
    device = logits.device
    target_lengths = torch.as_tensor([len(v['class_labels']) for v in targets], device=device)
    card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
    card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
    losses = {'cardinality_error': card_err}
    return losses