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
def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
    """Generate samples from the model and reference model for the given batch of inputs."""
    generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast
    with generate_context_manager():
        policy_output = model.generate(input_ids=batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        if 'reference_output' in batch:
            reference_output = batch['reference_output']
        elif self.ref_model is None:
            with self.null_ref_context():
                reference_output = self.model.generate(input_ids=batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        else:
            reference_output = self.ref_model.generate(input_ids=batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
    policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
    policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
    reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
    reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
    return (policy_output_decoded, reference_output_decoded)