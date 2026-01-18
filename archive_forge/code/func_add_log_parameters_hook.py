import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def add_log_parameters_hook(self, module: 'torch.nn.Module', name: str='', prefix: str='', log_freq: int=0) -> None:
    """This instruments hooks into the pytorch module
        log parameters after a forward pass
        log_freq - log gradients/parameters every N batches
        """
    prefix = prefix + name
    if not hasattr(module, '_wandb_hook_names'):
        module._wandb_hook_names = []

    def parameter_log_hook(module, input_, output, log_track):
        if not log_track_update(log_track):
            return
        for name, parameter in module.named_parameters():
            if isinstance(parameter, torch.autograd.Variable):
                data = parameter.data
            else:
                data = parameter
            self.log_tensor_stats(data.cpu(), 'parameters/' + prefix + name)
    log_track_params = log_track_init(log_freq)
    try:
        hook = module.register_forward_hook(lambda mod, inp, outp: parameter_log_hook(mod, inp, outp, log_track_params))
        self._hook_handles['parameters/' + prefix] = hook
        module._wandb_hook_names.append('parameters/' + prefix)
    except RuntimeError as e:
        wandb.termwarn(f'Trying to register forward_hook failed ({e}) - skipping parameter tracking.')