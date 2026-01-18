import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def open_for_append(self, fname):
    self._fname = fname
    logger.info('open: %s', fname)
    self._fp = open(fname, 'wb')