import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def frame_and_lineno(frame_or_tb: Union[FrameType, TracebackType]) -> Tuple[FrameType, int]:
    if is_frame(frame_or_tb):
        return (frame_or_tb, frame_or_tb.f_lineno)
    else:
        return (frame_or_tb.tb_frame, frame_or_tb.tb_lineno)