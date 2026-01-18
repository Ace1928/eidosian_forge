import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def iter_stack(frame_or_tb: Union[FrameType, TracebackType]) -> Iterator[Union[FrameType, TracebackType]]:
    current: Union[FrameType, TracebackType, None] = frame_or_tb
    while current:
        yield current
        if is_frame(current):
            current = current.f_back
        else:
            current = current.tb_next