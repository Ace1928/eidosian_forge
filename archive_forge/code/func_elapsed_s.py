from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
@property
def elapsed_s(self) -> str:
    """
        Returns the elapsed time since the timer was started as a string
        Does not add a checkpoint
        """
    return self.pformat(self.elapsed)