import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def save_audio(uri: InputType, src: torch.Tensor, sample_rate: int, channels_first: bool=True, format: Optional[str]=None, encoding: Optional[str]=None, bits_per_sample: Optional[int]=None, buffer_size: int=4096, compression: Optional[torchaudio.io.CodecConfig]=None) -> None:
    ext = None
    if hasattr(uri, 'write'):
        if format is None:
            raise RuntimeError("'format' is required when saving to file object.")
    else:
        uri = os.path.normpath(uri)
        if (tokens := str(uri).split('.')[1:]):
            ext = tokens[-1].lower()
    muxer, encoder, enc_fmt = _parse_save_args(ext, format, encoding, bits_per_sample)
    if channels_first:
        src = src.T
    s = torchaudio.io.StreamWriter(uri, format=muxer, buffer_size=buffer_size)
    s.add_audio_stream(sample_rate, num_channels=src.size(-1), format=_get_sample_format(src.dtype), encoder=encoder, encoder_format=enc_fmt, codec_config=compression)
    with s.open():
        s.write_audio_chunk(0, src)