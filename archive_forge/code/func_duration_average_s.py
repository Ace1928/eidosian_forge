from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def duration_average_s(self, count: int, checkpoint: Optional[bool]=False) -> str:
    """
        Returns the average duration of the timer as a string
        """
    return self.pformat(self.duration_average(count, checkpoint))