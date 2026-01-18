import inspect
import logging
import sys
import traceback
from collections import Counter
from html import escape as escape_html
from types import FrameType, TracebackType
from typing import Union, Iterable, List
from stack_data import (
from stack_data.utils import some_str
def format_traceback_part(self, e: BaseException) -> dict:
    return dict(frames=self.format_stack(e.__traceback__ or sys.exc_info()[2]), exception=dict(type=type(e).__name__, message=some_str(e)), tail='')