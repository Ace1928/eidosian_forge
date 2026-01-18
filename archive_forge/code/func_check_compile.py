import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .fake_tensor import fake_check
from torch._subclasses.schema_check_mode import SchemaCheckMode
def check_compile(func, args, kwargs, backend, fullgraph, dynamic):
    expected = func(*args, **kwargs)
    compiled_func = torch.compile(func, backend=backend, fullgraph=fullgraph, dynamic=dynamic)
    result = compiled_func(*args, **kwargs)
    msg = 'Output of func(*args, **kwargs) with or without torch.compile is different (under backend={backend}, dynamic={dynamic}). Given that the other tests have passed, this is likely a bug within the torch.compile stack.'
    torch.testing.assert_close(expected, result, msg=msg)