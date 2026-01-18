import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
def add_kivy_handlers(logger):
    """ Add Kivy-specific handlers to a logger.

    .. versionadded:: 2.2.0
    """
    logger.addHandler(LoggerHistory())
    if file_log_handler:
        logger.addHandler(file_log_handler)
    if sys.stderr and 'KIVY_NO_CONSOLELOG' not in os.environ:
        use_color = is_color_terminal()
        if not use_color:
            fmt = '[%(levelname)-7s] %(message)s'
        else:
            fmt = '[%(levelname)-18s] %(message)s'
        formatter = KivyFormatter(fmt, use_color=use_color)
        console = ConsoleHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)