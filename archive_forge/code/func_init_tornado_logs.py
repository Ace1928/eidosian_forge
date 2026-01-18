from __future__ import annotations
import logging
import sys
from typing import Final
def init_tornado_logs() -> None:
    """Set Tornado log levels.

    This function does not import any Tornado code, so it's safe to call even
    when Server is not running.
    """
    for log in ('access', 'application', 'general'):
        get_logger(f'tornado.{log}')