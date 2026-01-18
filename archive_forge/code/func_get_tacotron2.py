from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
@abstractmethod
def get_tacotron2(self, *, dl_kwargs=None) -> Tacotron2:
    """Create a Tacotron2 model with pre-trained weight.

        Args:
            dl_kwargs (dictionary of keyword arguments):
                Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Tacotron2:
                The resulting model.
        """