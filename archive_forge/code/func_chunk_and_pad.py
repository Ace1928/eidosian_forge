from enum import Enum
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence
import torch
import torch.distributed as dist
import torch.nn.functional as F
def chunk_and_pad(tensor: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
    """Chunk a given Tensor into num_chunks parts and add any necessary padding."""
    chunks = list(torch.flatten(tensor).chunk(num_chunks))
    num_pad_for_partial_chunk = chunks[0].numel() - chunks[-1].numel()
    if num_pad_for_partial_chunk > 0:
        chunks[-1] = F.pad(chunks[-1], [0, num_pad_for_partial_chunk])
    if len(chunks) < num_chunks:
        chunks.extend([torch.zeros_like(chunks[0]) for _ in range(num_chunks - len(chunks))])
    return chunks