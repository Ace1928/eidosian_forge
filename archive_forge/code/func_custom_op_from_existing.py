import dataclasses
import functools
import inspect
import sys
import typing
import weakref
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy
import torch
import torch._C as _C
import torch.library as library
from torch._library.abstract_impl import AbstractImplCtx
from torch.library import get_ctx
from .autograd import autograd_kernel_indirection, construct_autograd_kernel
def custom_op_from_existing(op):
    ns = op.namespace
    lib = torch.library.Library(ns, 'FRAGMENT')
    name = op.name().split('::')[-1]
    schema_str = str(op._schema)
    schema_str = schema_str.split('::')[-1]
    schema = FunctionSchema.parse(schema_str)
    return CustomOp(lib, ns, schema, name, op, _private_access=True)