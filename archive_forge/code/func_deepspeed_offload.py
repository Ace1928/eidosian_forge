import functools
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load
@property
def deepspeed_offload(self) -> bool:
    strategy = self.trainer.strategy
    if isinstance(strategy, DeepSpeedStrategy):
        cfg = strategy.config['zero_optimization']
        return cfg.get('offload_optimizer') or cfg.get('offload_param')
    return False