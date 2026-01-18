import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def scale_grad_by_num_grads_to_accum(self) -> None:
    """Scale the gradient down by the number of gradients to accumulate.

        This should be called after the gradient accumulation is done and the unscaled loss is used.
        """
    assert self._local_grad_sqr is None, 'Only call this after backward'
    assert self._num_grads_to_accum > 1, 'Must be accumulating gradients'
    assert not self._is_scaled_loss, 'Must use unscaled loss'
    for group in self._optimizer.param_groups:
        for param in group['params']:
            param.grad.div_(self._num_grads_to_accum)