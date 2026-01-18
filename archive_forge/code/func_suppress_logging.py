import logging
import logging.handlers
from collections import defaultdict
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import get_source_line
from sphinx.errors import SphinxWarning
from sphinx.util.console import colorize
from sphinx.util.osutil import abspath
@contextmanager
def suppress_logging() -> Generator[MemoryHandler, None, None]:
    """Context manager to suppress logging all logs temporarily.

    For example::

        >>> with suppress_logging():
        >>>     logger.warning('Warning message!')  # suppressed
        >>>     some_long_process()
        >>>
    """
    logger = logging.getLogger(NAMESPACE)
    memhandler = MemoryHandler()
    try:
        handlers = []
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handlers.append(handler)
        logger.addHandler(memhandler)
        yield memhandler
    finally:
        logger.removeHandler(memhandler)
        for handler in handlers:
            logger.addHandler(handler)