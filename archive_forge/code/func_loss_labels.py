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
def loss_labels(self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]) -> Dict[str, Tensor]:
    """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
    pred_logits = class_queries_logits
    batch_size, num_queries, _ = pred_logits.shape
    criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
    idx = self._get_predictions_permutation_indices(indices)
    target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
    target_classes = torch.full((batch_size, num_queries), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device)
    target_classes[idx] = target_classes_o
    pred_logits_transposed = pred_logits.transpose(1, 2)
    loss_ce = criterion(pred_logits_transposed, target_classes)
    losses = {'loss_cross_entropy': loss_ce}
    return losses