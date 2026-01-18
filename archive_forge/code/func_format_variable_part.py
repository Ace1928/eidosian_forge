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
def format_variable_part(self, text):
    if self.html:
        return escape_html(text)
    else:
        return text