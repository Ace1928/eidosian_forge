import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
@property
def phoneme_dict(self):
    """dict[str, tuple[str]]: Phonemes. Mapping from word to tuple of phonemes.
        Note that some words have empty phonemes.
        """
    if not self._phoneme_dict:
        self._phoneme_dict = {}
        with open(self._dict_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                content = line.strip().split()
                self._phoneme_dict[content[0]] = tuple(content[1:])
    return self._phoneme_dict.copy()