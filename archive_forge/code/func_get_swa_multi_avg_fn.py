import itertools
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def get_swa_multi_avg_fn():

    @torch.no_grad()
    def swa_update(averaged_param_list, current_param_list, num_averaged):
        if torch.is_floating_point(averaged_param_list[0]) or torch.is_complex(averaged_param_list[0]):
            torch._foreach_lerp_(averaged_param_list, current_param_list, 1 / (num_averaged + 1))
        else:
            diffs = torch._foreach_sub(current_param_list, averaged_param_list)
            torch._foreach_addcdiv_(averaged_param_list, diffs, [num_averaged + 1] * len(averaged_param_list))
    return swa_update