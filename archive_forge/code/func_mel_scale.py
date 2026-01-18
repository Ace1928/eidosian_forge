import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()