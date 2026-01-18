from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def record_python_call(self, fn: Callable, state: Mapping[str, str], message: Optional[str]=None, frames_to_skip: int=0) -> infra.ThreadFlowLocation:
    """Records a python call as one thread flow step."""
    frames_to_skip += 1
    stack = utils.python_call_stack(frames_to_skip=frames_to_skip, frames_to_log=5)
    location = utils.function_location(fn)
    location.message = message
    stack.frames.insert(0, infra.StackFrame(location=location))
    thread_flow_location = infra.ThreadFlowLocation(location=location, state=state, index=len(self.thread_flow_locations), stack=stack)
    self.with_thread_flow_location(thread_flow_location)
    return thread_flow_location