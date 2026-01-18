import copy
from typing import Any, cast
import srsly
from ..compat import mxnet as mx
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from ..util import (
from .shim import Shim
def finish_update(self, optimizer: Optimizer):
    params = []
    grads = []
    shapes = []
    ctx = mx.current_context()
    for key, value in self._model.collect_params().items():
        grad = cast(FloatsXd, mxnet2xp(value.grad(ctx)))
        param = cast(FloatsXd, mxnet2xp(value.data(ctx)))
        params.append(param.ravel())
        grads.append(grad.ravel())
        shapes.append((param.size, param.shape))
    if not params:
        return
    xp = get_array_module(params[0])
    flat_params, flat_grads = optimizer((self.id, 'mxnet-shim'), xp.concatenate(params), xp.concatenate(grads))
    start = 0
    for key, value in self._model.collect_params().items():
        size, shape = shapes.pop(0)
        param = flat_params[start:start + size].reshape(shape)
        value.set_data(xp2mxnet(param))
        value.zero_grad()
        start += size