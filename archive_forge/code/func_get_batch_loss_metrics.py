import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import (
def get_batch_loss_metrics(self, model, batch: Dict[str, Union[List, torch.LongTensor]], train_eval: Literal['train', 'eval']='train'):
    """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}
    policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = self.concatenated_forward(model, batch)
    if 'reference_chosen_logps' in batch and 'reference_rejected_logps' in batch:
        reference_chosen_logps = batch['reference_chosen_logps']
        reference_rejected_logps = batch['reference_rejected_logps']
    else:
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.model, batch)
            else:
                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.ref_model, batch)
    losses, chosen_rewards, rejected_rewards = self.dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    prefix = 'eval_' if train_eval == 'eval' else ''
    metrics[f'{prefix}rewards/chosen'] = chosen_rewards.mean().cpu()
    metrics[f'{prefix}rewards/rejected'] = rejected_rewards.mean().cpu()
    metrics[f'{prefix}rewards/accuracies'] = reward_accuracies.mean().cpu()
    metrics[f'{prefix}rewards/margins'] = (chosen_rewards - rejected_rewards).mean().cpu()
    metrics[f'{prefix}logps/rejected'] = policy_rejected_logps.detach().mean().cpu()
    metrics[f'{prefix}logps/chosen'] = policy_chosen_logps.detach().mean().cpu()
    metrics[f'{prefix}logits/rejected'] = policy_rejected_logits.detach().mean().cpu()
    metrics[f'{prefix}logits/chosen'] = policy_chosen_logits.detach().mean().cpu()
    return (losses.mean(), metrics)