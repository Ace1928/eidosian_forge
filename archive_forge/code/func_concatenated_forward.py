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
def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
    concatenated_batch = self.concatenated_inputs(batch, is_encoder_decoder=self.is_encoder_decoder, label_pad_token_id=self.label_pad_token_id, padding_value=self.padding_value, device=self.accelerator.device)
    len_chosen = batch['chosen_labels'].shape[0]
    model_kwargs = {'labels': concatenated_batch['concatenated_labels'], 'decoder_input_ids': concatenated_batch.pop('concatenated_decoder_input_ids', None)} if self.is_encoder_decoder else {}
    all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], use_cache=False, **model_kwargs).logits
    all_logps = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=self.loss_type == 'ipo', is_encoder_decoder=self.is_encoder_decoder, label_pad_token_id=self.label_pad_token_id)
    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]
    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]
    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)