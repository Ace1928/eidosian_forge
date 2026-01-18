import math
from typing import Callable, Optional, Protocol, Tuple
import torch
def logits_processor(logits: torch.Tensor) -> torch.Tensor:
    return logits / temperature