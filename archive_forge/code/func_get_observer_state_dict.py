import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def get_observer_state_dict(mod):
    """
    Returns the state dict corresponding to the observer stats.
    Traverse the model state_dict and extract out the stats.
    """
    od = OrderedDict()
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        for k, v in mod.state_dict().items():
            if 'observer' in k:
                od[k] = v
    else:
        for k, v in mod.state_dict().items():
            if 'activation_post_process' in k:
                od[k] = v
    od._metadata = mod.state_dict()._metadata
    return od