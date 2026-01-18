from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
def test_epoch_end(self, outputs):
    self.eval_epoch_end(outputs, prefix='test')