import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def in_last_block(self):
    """Determine if we're in the last block to handle in-progress writes."""
    return self._index > self._size_bytes - LEVELDBLOG_DATA_LEN