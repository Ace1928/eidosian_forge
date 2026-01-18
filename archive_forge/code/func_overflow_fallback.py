import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def overflow_fallback(self, y_int):
    """
        This fallback function is called when overflow is detected during training time, and adjusts the `self.shift`
        to avoid overflow in the subsequent runs.
        """
    self.set_shift(y_int)
    y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift)
    y_sq_int = y_int_shifted ** 2
    var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
    return var_int