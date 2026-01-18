from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def decode_bytes_batch(self, batch: list[list[int]], *, num_threads: int=8) -> list[bytes]:
    """Decodes a batch (list of lists of tokens) into a list of bytes."""
    with ThreadPoolExecutor(num_threads) as e:
        return list(e.map(self.decode_bytes, batch))