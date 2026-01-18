import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def schema_check(func, args, kwargs):
    with SchemaCheckMode():
        func(*args, **kwargs)