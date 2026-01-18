from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
@abstractmethod
def get_vocoder(self, *, dl_kwargs=None) -> Vocoder:
    """Create a vocoder module, based off of either WaveRNN or GriffinLim.

        If a pre-trained weight file is necessary,
        :func:`torch.hub.load_state_dict_from_url` is used to downloaded it.

        Args:
            dl_kwargs (dictionary of keyword arguments):
                Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Vocoder:
                A vocoder module, which takes spectrogram Tensor and an optional
                length Tensor, then returns resulting waveform Tensor and an optional
                length Tensor.
        """