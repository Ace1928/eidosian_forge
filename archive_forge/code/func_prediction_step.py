import inspect
import warnings
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from ..import_utils import is_peft_available
from .reward_config import RewardConfig
from .utils import RewardDataCollatorWithPadding, compute_accuracy
def prediction_step(self, model: Union[PreTrainedModel, nn.Module], inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]]=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    inputs = self._prepare_inputs(inputs)
    if ignore_keys is None:
        if hasattr(self.model, 'config'):
            ignore_keys = getattr(self.model.config, 'keys_to_ignore_at_inference', [])
        else:
            ignore_keys = []
    with torch.no_grad():
        loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)
    if prediction_loss_only:
        return (loss, None, None)
    loss = loss.detach()
    logits = tuple((v for k, v in logits_dict.items() if k not in ignore_keys))
    logits = nested_detach(logits)
    logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T
    labels = torch.zeros(logits.shape[0])
    labels = self._prepare_inputs(labels)
    return (loss, logits, labels)