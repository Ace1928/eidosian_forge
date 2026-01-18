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
def render_message(self, record: LogRecord, message: str) -> 'ConsoleRenderable':
    """Render message text in to Text.

        Args:
            record (LogRecord): logging Record.
            message (str): String containing log message.

        Returns:
            ConsoleRenderable: Renderable to display log message.
        """
    use_markup = getattr(record, 'markup', self.markup)
    message_text = Text.from_markup(message) if use_markup else Text(message)
    highlighter = getattr(record, 'highlighter', self.highlighter)
    if highlighter:
        message_text = highlighter(message_text)
    if self.keywords is None:
        self.keywords = self.KEYWORDS
    if self.keywords:
        message_text.highlight_words(self.keywords, 'logging.keyword')
    return message_text