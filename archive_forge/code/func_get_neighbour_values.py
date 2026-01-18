import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def get_neighbour_values(self, name, orig_val, radius=1, include_self=False):
    """
        Get neighbour values in 'radius' steps. The original value is not
        returned as it's own neighbour.
        """
    assert radius >= 1

    def update(cur_val, inc=True):
        if name == 'num_stages':
            if inc:
                return cur_val + 1
            else:
                return cur_val - 1
        elif inc:
            return cur_val * 2
        else:
            return cur_val // 2
    out = []
    cur_val = orig_val
    for _ in range(radius):
        cur_val = update(cur_val, True)
        if self.value_too_large(name, cur_val):
            break
        out.append(cur_val)
    cur_val = orig_val
    for _ in range(radius):
        cur_val = update(cur_val, False)
        if cur_val <= 0:
            break
        out.append(cur_val)
    if include_self:
        out.append(orig_val)
    return out