import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def split_by_batch_index(values, key, batch_idx):
    if key == 'scores':
        return [v[batch_idx].cpu() for v in values]
    if key == 'past_key_values':
        return None
    return values[batch_idx].cpu()