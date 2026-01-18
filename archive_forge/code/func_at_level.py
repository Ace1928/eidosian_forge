from contextlib import contextmanager
from contextlib import nullcontext
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import io
from io import StringIO
import logging
from logging import LogRecord
import os
from pathlib import Path
import re
from types import TracebackType
from typing import AbstractSet
from typing import Dict
from typing import final
from typing import Generator
from typing import Generic
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.capture import CaptureManager
from _pytest.config import _strtobool
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
@contextmanager
def at_level(self, level: Union[int, str], logger: Optional[str]=None) -> Generator[None, None, None]:
    """Context manager that sets the level for capturing of logs. After
        the end of the 'with' statement the level is restored to its original
        value.

        Will enable the requested logging level if it was disabled via :func:`logging.disable`.

        :param level: The level.
        :param logger: The logger to update. If not given, the root logger.
        """
    logger_obj = logging.getLogger(logger)
    orig_level = logger_obj.level
    logger_obj.setLevel(level)
    handler_orig_level = self.handler.level
    self.handler.setLevel(level)
    original_disable_level = self._force_enable_logging(level, logger_obj)
    try:
        yield
    finally:
        logger_obj.setLevel(orig_level)
        self.handler.setLevel(handler_orig_level)
        logging.disable(original_disable_level)