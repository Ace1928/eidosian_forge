import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
@property
def n_embed(self) -> int:
    return self._library.rwkv_get_n_embed(self._ctx)