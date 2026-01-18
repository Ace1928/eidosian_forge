from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchaudio._internal import load_state_dict_from_url
from torchaudio.prototype.models.hifi_gan import hifigan_vocoder, HiFiGANVocoder
from torchaudio.transforms import MelSpectrogram
def get_mel_transform(self) -> Module:
    """Construct an object which transforms waveforms into mel spectrograms."""
    return _HiFiGANMelSpectrogram(n_mels=self._vocoder_params['in_channels'], sample_rate=self._sample_rate, **self._mel_params)