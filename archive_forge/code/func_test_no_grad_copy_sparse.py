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
def test_no_grad_copy_sparse(self):

    class MyFunc(Function):
        static_grad_ptr = None

        @staticmethod
        def forward(ctx, inp):
            return inp

        @staticmethod
        def backward(ctx, grad):
            MyFunc.static_grad_ptr = grad._values().data_ptr()
            return grad

    class NonContGradFunc(Function):
        static_grad_ptr = None

        @staticmethod
        def forward(ctx, inp1, inp2):
            return inp1 + inp2

        @staticmethod
        def backward(ctx, grad):
            v = torch.rand(1, 3)
            i = torch.ones(1, 1, dtype=torch.long)
            nv = v.expand(8, 3)
            ni = i.expand(1, 8)
            ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
            NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
            return (ngrad, ngrad)
    a = torch.randn(10, 3, requires_grad=True)
    b = torch.randn(10, 3, requires_grad=True)
    input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
    offsets = torch.tensor([0, 4])
    import torch.nn.functional as F
    with dist_autograd.context() as context_id:
        emb_matrix = MyFunc.apply(a)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        dist_autograd.backward(context_id, [loss], retain_graph=True)
        grads = dist_autograd.get_gradients(context_id)
        p_g = MyFunc.static_grad_ptr
        p_a = grads[a]._values().data_ptr()
        self.assertTrue(p_a == p_g)
        for i in range(10):
            dist_autograd.backward(context_id, [loss], retain_graph=True)
    with dist_autograd.context() as context_id:
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        dist_autograd.backward(context_id, [loss], retain_graph=True)
        grads = dist_autograd.get_gradients(context_id)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = grads[a]._values().data_ptr()
        p_b = grads[b]._values().data_ptr()
        self.assertFalse(p_a == p_b)
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)
        for i in range(10):
            dist_autograd.backward(context_id, [loss], retain_graph=True)