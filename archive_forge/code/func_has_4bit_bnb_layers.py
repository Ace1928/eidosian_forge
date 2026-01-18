import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from accelerate.utils.imports import (
from ..big_modeling import dispatch_model, init_empty_weights
from .dataclasses import BnbQuantizationConfig
from .modeling import (
def has_4bit_bnb_layers(model):
    """Check if we have `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt` layers inside our model"""
    import bitsandbytes as bnb
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            return True
    return False