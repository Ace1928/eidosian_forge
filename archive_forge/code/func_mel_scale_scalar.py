import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)