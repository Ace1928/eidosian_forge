import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
@dist_init
def test_embedding_bag_with_no_grad_tensors(self):
    dst = self._next_rank()
    remote_embedding = rpc.remote(worker_name(dst), torch.nn.EmbeddingBag, args=(16, 16), kwargs={'mode': 'sum', 'sparse': True})
    local_embedding = torch.nn.EmbeddingBag(16, 16, mode='sum', sparse=True)
    input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
    per_sample_weights = torch.rand(8, requires_grad=True)
    offsets = torch.LongTensor([0, 4])
    local_res = local_embedding(input, offsets, per_sample_weights)
    torch.autograd.backward([local_res.sum()], retain_graph=True)
    torch.autograd.backward([local_res.sum()])
    local_grad = local_embedding.weight.grad
    with dist_autograd.context() as context_id:
        res = rpc.rpc_sync(worker_name(dst), DistAutogradTest._call_remote_embedding, args=(remote_embedding, input, offsets, per_sample_weights))
        dist_autograd.backward(context_id, [res.sum()], retain_graph=True)
        dist_autograd.backward(context_id, [res.sum()])
        remote_grad = rpc.rpc_sync(worker_name(dst), DistAutogradTest._get_grad, args=(remote_embedding, context_id))
        self.assertEqual(local_grad, remote_grad)