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
@staticmethod
def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]], is_encoder_decoder: bool=False, label_pad_token_id: int=-100, padding_value: int=0, device: Optional[torch.device]=None) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
    concatenated_batch = {}
    if is_encoder_decoder:
        max_length = max(batch['chosen_labels'].shape[1], batch['rejected_labels'].shape[1])
    else:
        max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            if 'labels' in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith('_input_ids'):
                pad_value = padding_value
            elif k.endswith('_attention_mask'):
                pad_value = 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            if 'labels' in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith('_input_ids'):
                pad_value = padding_value
            elif k.endswith('_attention_mask'):
                pad_value = 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((concatenated_batch[concatenated_key], pad_to_length(batch[k], max_length, pad_value=pad_value)), dim=0).to(device=device)
    if is_encoder_decoder:
        concatenated_batch['concatenated_input_ids'] = batch['prompt_input_ids'].repeat(2, 1).to(device=device)
        concatenated_batch['concatenated_attention_mask'] = batch['prompt_attention_mask'].repeat(2, 1).to(device=device)
    return concatenated_batch