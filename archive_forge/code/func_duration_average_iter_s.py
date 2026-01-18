from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def duration_average_iter_s(self, count: int, unit: Optional[str]=None, checkpoint: Optional[bool]=False) -> str:
    """
        Returns the average count/duration of the timer as a string
        """
    avg = self.duration_average_iter(count, checkpoint)
    return f'{avg:.2f}/sec' if unit is None else f'{avg:.2f} {unit}/sec'