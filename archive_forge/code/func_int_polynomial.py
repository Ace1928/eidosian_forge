import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def int_polynomial(self, x_int, scaling_factor):
    with torch.no_grad():
        b_int = torch.floor(self.coef[1] / scaling_factor)
        c_int = torch.floor(self.coef[2] / scaling_factor ** 2)
    z = (x_int + b_int) * x_int + c_int
    scaling_factor = self.coef[0] * scaling_factor ** 2
    return (z, scaling_factor)