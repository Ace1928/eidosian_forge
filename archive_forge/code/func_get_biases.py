import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
def get_biases(self):
    return (self.fc1.bias, self.fc2.bias)