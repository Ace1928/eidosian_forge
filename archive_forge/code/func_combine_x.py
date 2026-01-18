import argparse
import os
import tempfile
from typing import Tuple
import pandas as pd
import torch
import torch.nn as nn
import ray
import ray.train as train
from ray.data import Dataset
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer
def combine_x(batch):
    return pd.DataFrame({'x': batch[[f'x{i:03d}' for i in range(100)]].values.tolist(), 'y': batch['y']})