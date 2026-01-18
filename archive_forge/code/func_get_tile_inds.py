from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def get_tile_inds(format, device):
    transform = lambda x: F.transform(x.to(device), from_order='row', to_order=format)[0].to(x.device)
    with torch.no_grad():
        return get_inverse_transform_indices(transform, _get_tile_size(format)).to(device)