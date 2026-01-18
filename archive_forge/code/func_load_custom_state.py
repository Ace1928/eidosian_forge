import random
from pathlib import Path
from typing import List
import numpy as np
import torch
from safetensors.torch import load_file
from torch.cuda.amp import GradScaler
from .utils import (
from .logging import get_logger
from .state import PartialState
def load_custom_state(obj, path, index: int=0):
    """
    Loads the state of `obj` at `{path}/custom_checkpoint_{index}.pkl`
    """
    load_location = f'{path}/custom_checkpoint_{index}.pkl'
    logger.info(f'Loading the state of {get_pretty_name(obj)} from {load_location}')
    obj.load_state_dict(torch.load(load_location, map_location='cpu'))