import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def ensure_flushed(self, off: int) -> None:
    self._fp.flush()