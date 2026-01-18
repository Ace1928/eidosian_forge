import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
def mock_training(length, batch_size, generator, use_seedable_sampler=False):
    set_seed(42)
    generator.manual_seed(42)
    train_set = RegressionDataset(length=length, seed=42)
    train_dl = generate_baseline_dataloader(train_set, generator, batch_size, use_seedable_sampler)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch['x'])
            loss = torch.nn.functional.mse_loss(output, batch['y'])
            loss.backward()
            optimizer.step()
    return (train_set, model)