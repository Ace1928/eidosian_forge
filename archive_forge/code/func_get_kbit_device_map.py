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
def get_kbit_device_map() -> Optional[Dict[str, int]]:
    if is_xpu_available():
        return {'': f'xpu:{PartialState().local_process_index}'}
    elif torch.cuda.is_available():
        return {'': PartialState().local_process_index}
    else:
        return None