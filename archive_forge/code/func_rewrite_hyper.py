import argparse
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, cast
import pytorch_lightning as pl
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from xformers.benchmarks.LRA.code.dataset import LRADataset
from xformers.benchmarks.LRA.code.model_wrapper import ModelForSC, ModelForSCDual
from xformers.components.attention import ATTENTION_REGISTRY
def rewrite_hyper(config, rewrites):

    def replace(config_dict, k, v):
        if len(k.split(':')) == 1:
            config_dict[k] = v
            return
        first_key = k.split(':')[0]
        assert first_key in config_dict, first_key
        k = k[len(first_key) + 1:]
        replace(config_dict[first_key], k, v)
    for k, v in rewrites.items():
        replace(config, k, v)
    return config