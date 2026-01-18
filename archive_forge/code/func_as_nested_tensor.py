from typing import List, Optional, Union
import torch
from torch import SymInt, Tensor
from torch._C import _add_docstr, _nested  # type: ignore[attr-defined]
from torch.types import _device as Device, _dtype as DType
def as_nested_tensor(tensor_list: List[Tensor], dtype: Optional[DType]=None, device: Optional[Device]=None, layout=None) -> Tensor:
    """
    Constructs a nested tensor preserving autograd history from :attr:`tensor_list` a list of tensors.

    .. note::
        Tensors within the list are always copied by this function due to current nested tensor semantics.

    Args:
        tensor_list (List[Tensor]): a list of tensors with the same ndim

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.as_nested_tensor([a, b])
        >>> nt.is_leaf
        False
        >>> fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)])
        >>> nt.backward(fake_grad)
        >>> a.grad
        tensor([1., 1., 1.])
        >>> b.grad
        tensor([0., 0., 0., 0., 0.])
    """
    if not isinstance(tensor_list, list) or any((not isinstance(t, Tensor) for t in tensor_list)):
        raise TypeError('as_nested_tensor(): Expected first argument to be a list of tensors ')
    if layout is None:
        layout = torch.strided
    if layout == torch.strided:
        return torch._nested_tensor_from_tensor_list(tensor_list, dtype, None, device, None)
    elif layout == torch.jagged:
        from torch.nested._internal.nested_tensor import jagged_from_list
        nt, _ = jagged_from_list(tensor_list, offsets=None, device=device, dtype=dtype)
        return nt
    else:
        raise RuntimeError(f'Specified layout is unsupported for nested tensors: {layout}')