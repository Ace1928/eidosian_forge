from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def get_out_stream_info(self, i: int) -> OutputStreamTypes:
    """Get the metadata of output stream

        Args:
            i (int): Stream index.
        Returns:
            OutputStreamTypes
                Information about the output stream.
                If the output stream is audio type, then
                :class:`~torio.io._stream_reader.OutputAudioStream` is returned.
                If it is video type, then
                :class:`~torio.io._stream_reader.OutputVideoStream` is returned.
        """
    info = self._be.get_out_stream_info(i)
    return _parse_oi(info)