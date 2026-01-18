import weakref
import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._ops import OpOverload
from torch.library import Library
from torchgen.model import (
from .autograd import autograd_not_implemented
def register_functional_op(lib: Library, new_op_name: str, mutable_op: OpOverload) -> None:
    """Given a mutable operator, registers the functional variant.

    This API also correctly links the functional variant with the mutable
    operator for the purposes of functionalization.

    All of the new registrations are performed on the ``lib`` passed in.

    Arguments:
        lib (Library): Should be a torch.library.Library object that has
            the same namespace as ``mutable_op``'s namespace.
            lib will be used to register the new functional op as well
            as a functionalization kernel for the ``mutable_op``
            If you don't have a library handy, use
            ``torch.library.Library(ns, 'FRAGMENT')`` to construct one.
        new_op_name (str): The name of the functional operator (without the
            namespace). If no namespace, the new functional variant will be
            accessible under ``torch.ops.{lib.ns}.new_op_name``.
        mutable_op (OpOverload): The mutable custom operator. Note
            that you may need to add a `.default` to it, like
            `torch.ops.aten.abs_.default`.

    """
    validate(mutable_op)
    schema = functional_schema(new_op_name, mutable_op)
    lib.define(schema)
    functional_impl = construct_functional_impl(mutable_op)
    lib.impl(new_op_name, functional_impl, 'CompositeExplicitAutograd')
    functional_op = getattr(getattr(torch.ops, lib.ns), new_op_name).default
    lib.impl(new_op_name, autograd_not_implemented(functional_op), 'Autograd')
    f_kernel = construct_functionalization_kernel(weakref.proxy(mutable_op), functional_op)
    lib.impl(mutable_op, f_kernel, 'Functionalize')