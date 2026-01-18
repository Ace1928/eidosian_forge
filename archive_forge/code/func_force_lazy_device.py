import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
def force_lazy_device(model: fx.GraphModule):
    """
    Factory methods in a Fx graph may create tensors for a specific eager devices.
    If we take no actions, those eager tensors will be mixed with lazy tensors and
    cause crash. This method overwrite those eager device to lazy device.
    """

    def tolazydevice(dev):
        if isinstance(dev, torch.device):
            return torch.device('lazy', index=dev.index)
        return dev

    def hasDeviceArg(args, kwargs):
        return any((isinstance(arg, torch.device) for arg in itertools.chain(args, kwargs.values())))
    for nd in model.graph.nodes:
        nd.args = tuple((tolazydevice(arg) for arg in nd.args))
        nd.kwargs = {k: tolazydevice(v) for k, v in nd.kwargs.items()}
        if nd.target in tensor_factory_functions and (not hasDeviceArg(nd.args, nd.kwargs)):
            kwargs = dict(nd.kwargs)
            kwargs['device'] = torch.device('lazy')
            nd.kwargs = kwargs
    model.recompile()