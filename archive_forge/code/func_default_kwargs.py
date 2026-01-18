import collections
import logging
import torch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from .. import config, inductor_prims
from ..pattern_matcher import (
from ..virtualized import V
def default_kwargs(device):
    return {}