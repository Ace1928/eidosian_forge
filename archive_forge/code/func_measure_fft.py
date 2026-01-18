from typing import List
import torch
from torch import Tensor
def measure_fft(self, in_tensor: Tensor) -> List[Tensor]:
    """Like measure, but do it in FFT frequency domain."""
    return self.measure(torch.fft.fft(in_tensor, dim=self.dim).real)