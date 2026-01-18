import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset
def pad_block(block, pads):
    return torch.cat((pads.to(block.device), block), dim=-1).long()