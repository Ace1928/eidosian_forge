import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def remove_parametrizations(module: Module, tensor_name: str, leave_parametrized: bool=True) -> Module:
    """Remove the parametrizations on a tensor in a module.

    - If ``leave_parametrized=True``, ``module[tensor_name]`` will be set to
      its current output. In this case, the parametrization shall not change the ``dtype``
      of the tensor.
    - If ``leave_parametrized=False``, ``module[tensor_name]`` will be set to
      the unparametrised tensor in ``module.parametrizations[tensor_name].original``.
      This is only possible when the parametrization depends on just one tensor.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute :attr:`tensor_name` parametrized.
            Default: ``True``

    Returns:
        Module: module

    Raises:
        ValueError: if ``module[tensor_name]`` is not parametrized
        ValueError: if ``leave_parametrized=False`` and the parametrization depends on several tensors
    """
    if not is_parametrized(module, tensor_name):
        raise ValueError(f'Module {module} does not have a parametrization on {tensor_name}')
    assert isinstance(module.parametrizations, ModuleDict)
    parametrizations = module.parametrizations[tensor_name]
    if parametrizations.is_tensor:
        original = parametrizations.original
        if leave_parametrized:
            with torch.no_grad():
                t = getattr(module, tensor_name)
            with torch.no_grad():
                if type(original) is torch.Tensor:
                    original.set_(t)
                else:
                    try:
                        original.set_(t)
                    except RuntimeError as e:
                        raise RuntimeError('Calling remove_parametrizations() with leave_parametrized=True for a parameter that is an instance of a tensor subclass requires set_() to be implemented correctly for the tensor subclass. Either set leave_parametrized=False or provide a working implementation for set_() in the tensor subclass.') from e
    elif leave_parametrized:
        t = getattr(module, tensor_name)
        original = Parameter(t) if t.requires_grad else t
    else:
        raise ValueError('Cannot leave unparametrized (`leave_parametrized=False`) a tensor that is parametrized in terms of a sequence of tensors.')
    delattr(module.__class__, tensor_name)
    del module.parametrizations[tensor_name]
    _register_parameter_or_buffer(module, tensor_name, original)
    if not is_parametrized(module):
        delattr(module, 'parametrizations')
        orig_cls = module.__class__.__bases__[0]
        module.__class__ = orig_cls
    return module