import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def is_mha_input_projection(n):
    return 'q_proj' in n or 'k_proj' in n or 'v_proj' in n