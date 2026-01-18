import logging
import warnings
from functorch.compile import make_boxed_func
from ..backends.common import aot_autograd
from .registry import register_backend, register_experimental_backend
def openxla_eval_boxed(model, fake_tensor_inputs):
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)