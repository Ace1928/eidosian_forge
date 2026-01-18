import inspect
import functools
from enum import Enum
import torch.autograd
def profiler_record_fn_context(datapipe):
    if not hasattr(datapipe, '_profile_name'):
        datapipe._profile_name = _generate_iterdatapipe_msg(datapipe, simplify_dp_name=True)
    return torch.autograd.profiler.record_function(datapipe._profile_name)