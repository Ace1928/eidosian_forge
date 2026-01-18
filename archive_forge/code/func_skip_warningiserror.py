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
def skip_warningiserror(skip: bool=True) -> Generator[None, None, None]:
    """Context manager to skip WarningIsErrorFilter temporarily."""
    logger = logging.getLogger(NAMESPACE)
    if skip is False:
        yield
    else:
        try:
            disabler = DisableWarningIsErrorFilter()
            for handler in logger.handlers:
                handler.filters.insert(0, disabler)
            yield
        finally:
            for handler in logger.handlers:
                handler.removeFilter(disabler)