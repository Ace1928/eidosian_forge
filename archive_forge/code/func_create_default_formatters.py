from __future__ import annotations
import datetime
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.layout.dimension import AnyDimension, D
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.utils import get_cwidth
def create_default_formatters() -> list[Formatter]:
    """
    Return the list of default formatters.
    """
    return [Label(), Text(' '), Percentage(), Text(' '), Bar(), Text(' '), Progress(), Text(' '), Text('eta [', style='class:time-left'), TimeLeft(), Text(']', style='class:time-left'), Text(' ')]