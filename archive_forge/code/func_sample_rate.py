from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
@property
@abstractmethod
def sample_rate(self):
    """The sample rate of the resulting waveform

        :type: float
        """