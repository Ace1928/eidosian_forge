from __future__ import annotations
from types import FrameType
from typing import cast, Callable, Sequence
def should_start_context_test_function(frame: FrameType) -> str | None:
    """Is this frame calling a test_* function?"""
    co_name = frame.f_code.co_name
    if co_name.startswith('test') or co_name == 'runTest':
        return qualname_from_frame(frame)
    return None