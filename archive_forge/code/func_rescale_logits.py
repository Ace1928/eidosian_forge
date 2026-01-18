import math
from typing import Callable, Optional, Protocol, Tuple
import torch
def rescale_logits(temperature: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a function that rescales the token probabilities exponentially.

    Parameters
    ----------
    temperature
        The value by which we rescale the logits.

    """
    if not isinstance(temperature, float) or temperature < 0.0:
        raise ValueError(f'`temperature` must be a strictly negative floating point number, got {temperature} instead.')
    elif temperature == 0.0:
        raise ValueError('Please use the greedy sampler instead of setting the temperature to 0.')

    def logits_processor(logits: torch.Tensor) -> torch.Tensor:
        return logits / temperature
    return logits_processor