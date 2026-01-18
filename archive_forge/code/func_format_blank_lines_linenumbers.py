import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def format_blank_lines_linenumbers(self, blank_line):
    if self.current_line_indicator:
        result = ' ' * len(self.current_line_indicator) + ' '
    else:
        result = '   '
    if blank_line.begin_lineno == blank_line.end_lineno:
        return result + self.line_number_format_string.format(blank_line.begin_lineno) + '\n'
    return result + '   {}\n'.format(self.line_number_gap_string)