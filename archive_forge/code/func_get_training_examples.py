import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
def get_training_examples():
    n = 16
    training_examples = FeatureSet(dense_features=torch.zeros((n, D_DENSE)), sparse_features=torch.zeros(n, dtype=torch.long), values=torch.zeros(n))
    idx = 0
    for value in (-1, 1):
        for x in (-1.0 * value, 1.0 * value):
            for y in (1.0 * value, -1.0 * value):
                for z in (0, 1):
                    training_examples.dense_features[idx, :] = torch.tensor((x, y))
                    training_examples.sparse_features[idx] = z
                    training_examples.values[idx] = value
                    idx += 1
    assert 0 == n % NUM_TRAINERS
    examples_per_trainer = int(n / NUM_TRAINERS)
    return [FeatureSet(dense_features=training_examples.dense_features[start:start + examples_per_trainer, :], sparse_features=training_examples.sparse_features[start:start + examples_per_trainer], values=training_examples.values[start:start + examples_per_trainer]) for start in range(0, n, examples_per_trainer)]