import datetime as dt
import logging
import platform
import threading
import time
import uuid
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import pandas
import psutil
import modin
from modin.config import LogFileSize, LogMemoryInterval, LogMode
def bytes_int_to_str(num_bytes: int, suffix: str='B') -> str:
    """
    Scale bytes to its human-readable format (e.g: 1253656678 => '1.17GB').

    Parameters
    ----------
    num_bytes : int
        Number of bytes.
    suffix : str, default: "B"
        Suffix to add to conversion of num_bytes.

    Returns
    -------
    str
        Human-readable string format.
    """
    factor = 1000
    n_bytes: float = num_bytes
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if n_bytes < factor:
            return f'{n_bytes:.2f}{unit}{suffix}'
        n_bytes /= factor
    return f'{n_bytes * 1000:.2f}P{suffix}'