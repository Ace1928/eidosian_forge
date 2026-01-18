import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the frequency tensor for broadcasting over the input tensor.

    This function adjusts the shape of the frequency tensor so that it can be broadcasted over the input tensor for element-wise operations.

    Parameters:
        freqs_cis (torch.Tensor): The frequency tensor.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The reshaped frequency tensor ready for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, 'Tensor x must have at least two dimensions.'
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), 'Mismatch in shapes for broadcasting.'
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)