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
def test_no_grad_copy(self):
    """
        Similar to test in test_autograd.py.
        """

    class MyFunc(Function):
        static_grad_ptr = None

        @staticmethod
        def forward(ctx, inp1, inp2):
            return inp1 + inp2

        @staticmethod
        def backward(ctx, grad):
            MyFunc.static_grad_ptr = grad.data_ptr()
            return (grad, grad)

    class MyFuncSingleGrad(Function):
        static_grad_ptr = None

        @staticmethod
        def forward(ctx, inp):
            return inp

        @staticmethod
        def backward(ctx, grad):
            MyFuncSingleGrad.static_grad_ptr = grad.data_ptr()
            return grad

    class NonContGradFunc(Function):

        @staticmethod
        def forward(ctx, inp1):
            ctx.size = inp1.size()
            return torch.tensor([1.0])

        @staticmethod
        def backward(ctx, grad):
            return torch.ones(1).expand(ctx.size)
    a = torch.randn(5, 6, requires_grad=True)
    b = torch.randn(5, 6, requires_grad=True)
    with dist_autograd.context() as context_id:
        dist_autograd.backward(context_id, [NonContGradFunc.apply(MyFunc.apply(a, b))])
        grads = dist_autograd.get_gradients(context_id)
        self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)
    with dist_autograd.context() as context_id:
        dist_autograd.backward(context_id, [MyFuncSingleGrad.apply(a)[1][0]])
        grads = dist_autograd.get_gradients(context_id)
        p_g = MyFuncSingleGrad.static_grad_ptr
        p_a = grads[a].data_ptr()
        self.assertTrue(p_a == p_g)
    with dist_autograd.context() as context_id:
        dist_autograd.backward(context_id, [MyFunc.apply(a, b)[1][0]])
        grads = dist_autograd.get_gradients(context_id)
        p_g = MyFunc.static_grad_ptr
        p_a = grads[a].data_ptr()
        p_b = grads[b].data_ptr()
        self.assertFalse(p_a == p_b)
        self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)