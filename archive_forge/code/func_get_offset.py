import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def get_offset(self) -> int:
    offset = self._fp.tell()
    return offset