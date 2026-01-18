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
def pending_logging() -> Generator[MemoryHandler, None, None]:
    """Context manager to postpone logging all logs temporarily.

    For example::

        >>> with pending_logging():
        >>>     logger.warning('Warning message!')  # not flushed yet
        >>>     some_long_process()
        >>>
        Warning message!  # the warning is flushed here
    """
    logger = logging.getLogger(NAMESPACE)
    try:
        with suppress_logging() as memhandler:
            yield memhandler
    finally:
        memhandler.flushTo(logger)