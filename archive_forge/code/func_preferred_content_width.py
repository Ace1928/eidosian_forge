from __future__ import annotations
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence, Union, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth, take_using_weights, to_int, to_str
from .controls import (
from .dimension import (
from .margins import Margin
from .mouse_handlers import MouseHandlers
from .screen import _CHAR_CACHE, Screen, WritePosition
from .utils import explode_text_fragments
def preferred_content_width() -> int | None:
    """Content width: is only calculated if no exact width for the
            window was given."""
    if self.ignore_content_width():
        return None
    total_margin_width = self._get_total_margin_width()
    preferred_width = self.content.preferred_width(max_available_width - total_margin_width)
    if preferred_width is not None:
        preferred_width += total_margin_width
    return preferred_width