import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def format_variable(self, var: Variable) -> str:
    return '{} = {}'.format(var.name, self.format_variable_value(var.value))