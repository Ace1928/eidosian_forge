import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from accelerate import PartialState
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from ..import_utils import is_peft_available, is_unsloth_available, is_xpu_available
from ..trainer.model_config import ModelConfig
def get_peft_config(model_config: ModelConfig) -> 'Optional[PeftConfig]':
    if model_config.use_peft is False:
        return None
    peft_config = LoraConfig(r=model_config.lora_r, lora_alpha=model_config.lora_alpha, lora_dropout=model_config.lora_dropout, bias='none', task_type='CAUSAL_LM', target_modules=model_config.lora_target_modules, modules_to_save=model_config.lora_modules_to_save)
    return peft_config