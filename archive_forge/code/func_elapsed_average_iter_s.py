from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def elapsed_average_iter_s(self, count: int, unit: Optional[str]=None) -> str:
    """
        Returns the average count/elapsed duration of the timer as a string
        """
    avg = self.elapsed_average_iter(count)
    return f'{avg:.2f}/sec' if unit is None else f'{avg:.2f} {unit}/sec'