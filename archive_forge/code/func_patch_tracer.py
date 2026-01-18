import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import torch
import torch.nn as nn
@contextmanager
def patch_tracer(self, tracer: torch.fx.Tracer, root_module: nn.Module):
    self.exec_info = _ExecutionInfo(root_module)
    orig_call_module = tracer.call_module
    orig_create_proxy = tracer.create_proxy
    tracer.call_module = functools.partial(self._patched_call_module, orig_call_module, self.exec_info)
    fqn_to_param = dict(root_module.named_parameters())
    tracer.create_proxy = functools.partial(self._patched_create_proxy, orig_create_proxy, self.exec_info, fqn_to_param)
    try:
        yield
    finally:
        tracer.call_module = orig_call_module
        tracer.create_proxy = orig_create_proxy