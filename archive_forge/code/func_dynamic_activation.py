import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import ActivationDictionary
from IndegoLogging import configure_logging
def dynamic_activation(self, x: torch.Tensor, batch_id: Optional[int], deterministic: bool=False) -> torch.Tensor:
    """
        Dynamically selects and applies activation functions based on the policy network's output,
        with an option for deterministic or probabilistic selection of activations.

        Args:
            x (torch.Tensor): The input tensor to be processed through activation functions.
            batch_id (Optional[int]): An identifier for the batch, used for logging purposes.
            deterministic (bool): If True, the activation function with the highest score is selected,
                                  otherwise, a probabilistic approach is used.

        Returns:
            torch.Tensor: The tensor resulting from applying the selected activation functions,
                          with padding applied to ensure all output tensors have the same size.
        """
    activation_scores: torch.Tensor = self.policy_network(x)
    logger.debug(f'Activation scores computed for batch_id={batch_id}: {activation_scores}')
    if deterministic:
        selected_activation_idx: torch.Tensor = torch.argmax(activation_scores, dim=-1)
        logger.info(f'Deterministic selection of activation indices for batch_id={batch_id}: {selected_activation_idx}')
    else:
        activation_probs: torch.Tensor = F.softmax(activation_scores, dim=-1)
        selected_activation_idx: torch.Tensor = torch.multinomial(activation_probs, 1).squeeze(0)
        logger.info(f'Probabilistic selection of activation indices for batch_id={batch_id}: {selected_activation_idx}')
    selected_activations: List[Callable[[torch.Tensor], torch.Tensor]] = [self.activations[self.activation_keys[int(idx.long())]] for idx in selected_activation_idx.long()]
    logger.debug(f'Selected activation functions for batch_id={batch_id}: {selected_activations}')
    activated_tensors: List[torch.Tensor] = [act(x_i) for x_i, act in zip(x, selected_activations)]
    logger.debug(f'Activated tensors before padding for batch_id={batch_id}: {activated_tensors}')
    max_size: int = max((tensor.nelement() for tensor in activated_tensors), default=0)
    padded_tensors: List[torch.Tensor] = []
    for tensor in activated_tensors:
        if tensor.nelement() == 0:
            padded_tensor = torch.zeros((max_size,), dtype=tensor.dtype, device=tensor.device) if max_size > 0 else torch.tensor([], dtype=tensor.dtype, device=tensor.device)
        else:
            padding_needed = max_size - tensor.nelement()
            padded_tensor = F.pad(tensor, (0, padding_needed), 'constant', 0)
        padded_tensors.append(padded_tensor)
        logger.debug(f'Padded tensor for batch_id={batch_id}: {padded_tensor}')
    final_output = torch.stack(padded_tensors)
    logger.debug(f'Final output tensor after dynamic activation for batch_id={batch_id}: {final_output.shape}')
    return final_output