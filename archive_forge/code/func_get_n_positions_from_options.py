import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def get_n_positions_from_options(opt):
    """
    Determine n_positions from options dict.
    """
    if opt.get('n_positions'):
        n_positions = opt['n_positions']
    else:
        n_positions = max(opt.get('truncate') or 0, opt.get('text_truncate') or 0, opt.get('label_truncate') or 0)
        if n_positions == 0:
            n_positions = 1024
    return n_positions