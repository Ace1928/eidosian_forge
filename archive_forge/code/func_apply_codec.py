import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
@deprecated('Please migrate to :py:class:`torchaudio.io.AudioEffector`.', remove=False)
def apply_codec(waveform: Tensor, sample_rate: int, format: str, channels_first: bool=True, compression: Optional[float]=None, encoding: Optional[str]=None, bits_per_sample: Optional[int]=None) -> Tensor:
    """
    Apply codecs as a form of augmentation.

    .. devices:: CPU

    Args:
        waveform (Tensor): Audio data. Must be 2 dimensional. See also ```channels_first```.
        sample_rate (int): Sample rate of the audio waveform.
        format (str): File format.
        channels_first (bool, optional):
            When True, both the input and output Tensor have dimension `(channel, time)`.
            Otherwise, they have dimension `(time, channel)`.
        compression (float or None, optional): Used for formats other than WAV.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.
        encoding (str or None, optional): Changes the encoding for the supported formats.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.
        bits_per_sample (int or None, optional): Changes the bit depth for the supported formats.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.

    Returns:
        Tensor: Resulting Tensor.
        If ``channels_first=True``, it has `(channel, time)` else `(time, channel)`.
    """
    from torchaudio.backend import _sox_io_backend
    with tempfile.NamedTemporaryFile() as f:
        torchaudio.backend._sox_io_backend.save(f.name, waveform, sample_rate, channels_first, compression, format, encoding, bits_per_sample)
        augmented, sr = _sox_io_backend.load(f.name, channels_first=channels_first, format=format)
    if sr != sample_rate:
        augmented = resample(augmented, sr, sample_rate)
    return augmented