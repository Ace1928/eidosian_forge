import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def logger_warning(msg: str, src: str) -> None:
    """
    Use this instead of logger.warning directly.

    That allows people to overwrite it more easily.

    ## Exception, warnings.warn, logger_warning
    - Exceptions should be used if the user should write code that deals with
      an error case, e.g. the PDF being completely broken.
    - warnings.warn should be used if the user needs to fix their code, e.g.
      DeprecationWarnings
    - logger_warning should be used if the user needs to know that an issue was
      handled by pypdf, e.g. a non-compliant PDF being read in a way that
      pypdf could apply a robustness fix to still read it. This applies mainly
      to strict=False mode.
    """
    logging.getLogger(src).warning(msg)