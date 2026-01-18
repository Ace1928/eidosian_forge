from functools import update_wrapper
from numbers import Number
from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch.overrides import is_tensor_like
def broadcast_all(*values):
    """
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all((is_tensor_like(v) or isinstance(v, Number) for v in values)):
        raise ValueError('Input arguments must all be instances of numbers.Number, torch.Tensor or objects implementing __torch_function__.')
    if not all((is_tensor_like(v) for v in values)):
        options: Dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [v if is_tensor_like(v) else torch.tensor(v, **options) for v in values]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)