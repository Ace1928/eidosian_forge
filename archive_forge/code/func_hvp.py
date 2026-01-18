from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def hvp(func, inputs, v=None, create_graph=False, strict=False):
    """Compute the dot product between the scalar function's Hessian and a vector ``v`` at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            hvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> hvp(pow_reducer, inputs, v)
        (tensor(0.1448),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]]))

        >>> hvp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.1448, grad_fn=<SumBackward0>),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> hvp(pow_adder_reducer, inputs, v)
        (tensor(2.3030),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    Note:

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your functions is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, 'inputs', 'hvp')
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
        if v is not None:
            _, v = _as_tuple(v, 'v', 'hvp')
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        elif len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError('The vector v can only be None if the input to the user-provided function is a single Tensor with a single element.')
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, 'outputs of the user-provided function', 'hvp')
        _check_requires_grad(outputs, 'outputs', strict=strict)
        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError('The function given to hvp should return a single Tensor')
        if outputs[0].nelement() != 1:
            raise RuntimeError('The Tensor returned by the function given to hvp should contain a single element')
        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, 'jacobian', strict=strict)
        grad_jac = tuple((torch.zeros_like(inp, requires_grad=True) for inp in inputs))
        double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
        _check_requires_grad(jac, 'hessian', strict=strict)
    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)
        hvp = _fill_in_zeros(grad_res, inputs, strict, create_graph, 'double_back_trick')
    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)
    return (_tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(hvp, is_inputs_tuple))