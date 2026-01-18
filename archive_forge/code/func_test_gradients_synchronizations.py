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
@skip_if_lt_x_gpu(4)
def test_gradients_synchronizations(self):
    options = self.rpc_backend_options
    for peer_rank in range(self.world_size):
        options.set_device_map(worker_name(peer_rank), {self.rank: peer_rank})
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    if self.rank == 0:
        layers = [nn.Linear(2000, 2000) for _ in range(self.world_size - 1)]
        local_layers = [l.to(0) for l in layers]
        remote_layers = []
        for rank in range(1, self.world_size):
            remote_layers.append(rpc.remote(worker_name(rank), WrapperModule, args=(layers[rank - 1], rank)))
        x = torch.randn(5000, 2000).to(0)
        local_model = nn.Sequential(*local_layers)
        local_model(x).sum().backward()
        with dist_autograd.context() as context_id:
            for remote_layer in remote_layers:
                x = remote_layer.rpc_sync().forward(x)
            dist_autograd.backward(context_id, [x.sum()])
            futs = []
            for remote_layer in remote_layers:
                futs.append(remote_layer.rpc_async().gradients(context_id))
            for i in range(len(futs)):
                local_gradients = [p.grad for p in local_layers[i].parameters()]
                for g1, g2 in zip(futs[i].wait(), local_gradients):
                    self.assertEqual(g1, g2)
    rpc.shutdown()