import itertools
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def get_swa_avg_fn():

    @torch.no_grad()
    def swa_update(averaged_param, current_param, num_averaged):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)
    return swa_update