from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def pop_chunks(self) -> Tuple[Optional[ChunkTensor]]:
    """Pop one chunk from all the output stream buffers.

        Returns:
            Tuple[Optional[ChunkTensor]]:
                Buffer contents.
                If a buffer does not contain any frame, then `None` is returned instead.
        """
    ret = []
    for chunk in self._be.pop_chunks():
        if chunk is None:
            ret.append(None)
        else:
            ret.append(ChunkTensor(chunk.frames, chunk.pts))
    return ret