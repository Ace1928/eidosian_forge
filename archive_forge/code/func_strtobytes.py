import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def strtobytes(x):
    """strtobytes."""
    return bytes(x, 'iso8859-1')