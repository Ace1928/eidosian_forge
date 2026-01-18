import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
def get_bias_grads(self):
    return (self.fc1.bias.grad, self.fc2.bias.grad)