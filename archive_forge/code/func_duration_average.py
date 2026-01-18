from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def duration_average(self, count: int, checkpoint: Optional[bool]=False) -> float:
    """
        Returns the average duration of the timer
        """
    return self.get_duration(checkpoint) / count