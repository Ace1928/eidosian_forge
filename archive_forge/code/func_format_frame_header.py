import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def format_frame_header(self, frame_info: FrameInfo) -> str:
    return ' File "{frame_info.filename}", line {frame_info.lineno}, in {name}\n'.format(frame_info=frame_info, name=frame_info.executing.code_qualname() if self.use_code_qualname else frame_info.code.co_name)