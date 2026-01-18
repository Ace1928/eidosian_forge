import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def parameter_log_hook(module, input_, output, log_track):
    if not log_track_update(log_track):
        return
    for name, parameter in module.named_parameters():
        if isinstance(parameter, torch.autograd.Variable):
            data = parameter.data
        else:
            data = parameter
        self.log_tensor_stats(data.cpu(), 'parameters/' + prefix + name)