import itertools
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def get_ema_multi_avg_fn(decay=0.999):

    @torch.no_grad()
    def ema_update(ema_param_list, current_param_list, _):
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(ema_param_list[0]):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))
    return ema_update