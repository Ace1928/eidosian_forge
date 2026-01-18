import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
@property
def arch_version_major(self) -> int:
    return self._library.rwkv_get_arch_version_major(self._ctx)