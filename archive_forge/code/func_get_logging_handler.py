import logging
from typing import Dict
def get_logging_handler(destination: str='null') -> logging.Handler:
    global _log_handlers
    return _log_handlers[destination]