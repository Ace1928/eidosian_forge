import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init()
def test_dist_optim(self):
    self._test_dist_optim_base(optim.Adagrad, lr=0.05)
    self._test_dist_optim_base(optim.Adam, lr=0.01, amsgrad=True)
    self._test_dist_optim_base(optim.AdamW, lr=0.05, amsgrad=True)
    self._test_dist_optim_base(optim.SGD, lr=0.05)
    self._test_dist_optim_base(optim.SGD, lr=0.001, momentum=1, weight_decay=1, nesterov=True)
    self._test_dist_optim_base(optim.Adadelta, rho=0.95)
    self._test_dist_optim_base(optim.RMSprop, lr=0.05)
    self._test_dist_optim_base(optim.Adamax, lr=0.05)
    self._test_dist_optim_base(optim.Rprop, lr=0.05)