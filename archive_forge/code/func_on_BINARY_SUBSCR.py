from _pydev_bundle import pydev_log
from types import CodeType
from _pydevd_frame_eval.vendored.bytecode.instr import _Variable
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import cfg as bytecode_cfg
import dis
import opcode as _opcode
from _pydevd_bundle.pydevd_constants import KeyifyList, DebugInfoHolder, IS_PY311_OR_GREATER
from bisect import bisect
from collections import deque
def on_BINARY_SUBSCR(self, instr):
    try:
        _sub = self._stack.pop()
    except IndexError:
        return
    try:
        _container = self._stack.pop()
    except IndexError:
        return
    self.function_calls.append(Target(_BINARY_OP_MAP[instr.name], instr.lineno, instr.offset))
    self._stack.append(instr)