import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
@functools.lru_cache
def namedtuple_args_cls(schema):
    attribs = [arg.name for arg in schema.arguments.flat_all]
    name = str(schema.name) + '_args'
    tuple_cls = namedtuple(name, attribs)
    return tuple_cls