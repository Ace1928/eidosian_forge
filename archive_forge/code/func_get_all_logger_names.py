import asyncio
import atexit
import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, List, Optional, Tuple, Union
def get_all_logger_names(include_root: bool=False) -> List[str]:
    """Return ``list`` of names of all loggers than have been accessed.

    Warning: this is sensitive to internal structures in the standard logging module.
    """
    rv = list(logging.Logger.manager.loggerDict.keys())
    if include_root:
        rv.insert(0, '')
    return rv