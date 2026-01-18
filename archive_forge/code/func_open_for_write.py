import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def open_for_write(self, fname: str) -> None:
    self._fname = fname
    logger.info('open: %s', fname)
    open_flags = 'xb'
    self._fp = open(fname, open_flags)
    self._write_header()