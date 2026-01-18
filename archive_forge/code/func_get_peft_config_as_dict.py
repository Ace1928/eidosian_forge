from __future__ import annotations
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
def get_peft_config_as_dict(self, inference: bool=False):
    config_dict = {}
    for key, value in self.peft_config.items():
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
        if inference:
            config['inference_mode'] = True
    config_dict[key] = config
    return config