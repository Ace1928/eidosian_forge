import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def hook_torch_modules(self, module, criterion=None, prefix=None, graph_idx=0, parent=None):
    torch = util.get_module('torch', 'Could not import torch')
    layers = 0
    graph = self
    if hasattr(module, '_wandb_watch_called') and module._wandb_watch_called:
        raise ValueError('You can only call `wandb.watch` once per model.  Pass a new instance of the model if you need to call wandb.watch again in your code.')
    module._wandb_watch_called = True
    if criterion:
        graph.criterion = criterion
        graph.criterion_passed = True
    for name, sub_module in module.named_children():
        name = name or str(layers)
        if prefix:
            name = prefix + '.' + name
        layers += 1
        if not isinstance(sub_module, torch.nn.Module):
            break
        module_types = [getattr(torch.nn, module_classname) for module_classname in ('Container', 'Sequential', 'ModuleList', 'ModuleDict') if hasattr(torch.nn, module_classname)]
        if parent is None:
            parent = module
        if isinstance(sub_module, tuple(module_types)):
            self.hook_torch_modules(sub_module, prefix=name, parent=parent)
        else:
            self._graph_hooks |= {id(sub_module)}
            try:
                graph_hook = sub_module.register_forward_hook(self.create_forward_hook(name, graph_idx))
                wandb.run._torch._hook_handles['topology/' + str(id(graph_hook))] = graph_hook
                if not hasattr(parent, '_wandb_hook_names'):
                    parent._wandb_hook_names = []
                parent._wandb_hook_names.append('topology/' + str(id(graph_hook)))
            except RuntimeError as e:
                wandb.termwarn(f'Trying to register forward_hook failed ({e}) - skipping graph tracking.', repeat=False)