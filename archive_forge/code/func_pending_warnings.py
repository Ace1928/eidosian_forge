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
def pending_warnings() -> Generator[logging.Handler, None, None]:
    """Context manager to postpone logging warnings temporarily.

    Similar to :func:`pending_logging`.
    """
    logger = logging.getLogger(NAMESPACE)
    memhandler = MemoryHandler()
    memhandler.setLevel(logging.WARNING)
    try:
        handlers = []
        for handler in logger.handlers[:]:
            if isinstance(handler, WarningStreamHandler):
                logger.removeHandler(handler)
                handlers.append(handler)
        logger.addHandler(memhandler)
        yield memhandler
    finally:
        logger.removeHandler(memhandler)
        for handler in handlers:
            logger.addHandler(handler)
        memhandler.flushTo(logger)