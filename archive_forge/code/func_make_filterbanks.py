from typing import Optional
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
def make_filterbanks(n_fft=4096, n_hop=1024, center=False, sample_rate=44100.0, method='torch'):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
    if method == 'torch':
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == 'asteroid':
        fb = torch_stft_fb.TorchSTFTFB.from_torch_args(n_fft=n_fft, hop_length=n_hop, win_length=n_fft, window=window, center=center, sample_rate=sample_rate)
        encoder = AsteroidSTFT(fb)
        decoder = AsteroidISTFT(fb)
    else:
        raise NotImplementedError
    return (encoder, decoder)