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
def save_custom_state(obj, path, index: int=0, save_on_each_node: bool=False):
    """
    Saves the state of `obj` to `{path}/custom_checkpoint_{index}.pkl`
    """
    save_location = Path(path) / f'custom_checkpoint_{index}.pkl'
    logger.info(f'Saving the state of {get_pretty_name(obj)} to {save_location}')
    save(obj.state_dict(), save_location, save_on_each_node=save_on_each_node)