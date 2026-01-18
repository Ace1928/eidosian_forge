import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def int_erf(self, x_int, scaling_factor):
    b_int = torch.floor(self.coeff[1] / scaling_factor)
    c_int = torch.floor(self.coeff[2] / scaling_factor ** 2)
    sign = torch.sign(x_int)
    abs_int = torch.min(torch.abs(x_int), -b_int)
    y_int = sign * ((abs_int + b_int) ** 2 + c_int)
    scaling_factor = scaling_factor ** 2 * self.coeff[0]
    y_int = floor_ste.apply(y_int / 2 ** self.const)
    scaling_factor = scaling_factor * 2 ** self.const
    return (y_int, scaling_factor)