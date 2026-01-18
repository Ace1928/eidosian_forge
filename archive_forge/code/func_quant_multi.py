from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
def quant_multi(x, dim):
    max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    max1[max1 == 0] = 1.0
    x = torch.round(x / max1 * 127)
    return (max1, x.to(torch.int8))