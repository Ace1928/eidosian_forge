import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)