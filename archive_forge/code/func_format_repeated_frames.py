import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def format_repeated_frames(self, repeated_frames: RepeatedFrames) -> str:
    return '    [... skipping similar frames: {}]\n'.format(repeated_frames.description)