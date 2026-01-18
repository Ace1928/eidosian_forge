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
def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
    """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
    compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
    with torch.no_grad(), compte_ref_context_manager():
        if self.ref_model is None:
            with self.null_ref_context():
                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.model, padded_batch)
        else:
            reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.ref_model, padded_batch)
    return (reference_chosen_logps, reference_rejected_logps)