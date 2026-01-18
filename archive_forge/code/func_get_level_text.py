import logging
from datetime import datetime
from logging import Handler, LogRecord
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Iterable, List, Optional, Type, Union
from pip._vendor.rich._null_file import NullFile
from . import get_console
from ._log_render import FormatTimeCallable, LogRender
from .console import Console, ConsoleRenderable
from .highlighter import Highlighter, ReprHighlighter
from .text import Text
from .traceback import Traceback
def get_level_text(self, record: LogRecord) -> Text:
    """Get the level name from the record.

        Args:
            record (LogRecord): LogRecord instance.

        Returns:
            Text: A tuple of the style and level name.
        """
    level_name = record.levelname
    level_text = Text.styled(level_name.ljust(8), f'logging.level.{level_name.lower()}')
    return level_text