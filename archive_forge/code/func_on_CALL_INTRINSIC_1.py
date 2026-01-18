from _pydev_bundle import pydev_log
from types import CodeType
from _pydevd_frame_eval.vendored.bytecode.instr import _Variable, Label
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import cfg as bytecode_cfg
import dis
import opcode as _opcode
from _pydevd_bundle.pydevd_constants import KeyifyList, DebugInfoHolder, IS_PY311_OR_GREATER
from bisect import bisect
from collections import deque
import traceback
def on_CALL_INTRINSIC_1(self, instr):
    try:
        func_name_instr = self._stack.pop()
    except IndexError:
        return
    if self._stack:
        peeked = self._stack[-1]
        if peeked.name == 'PUSH_NULL':
            self._stack.pop()
    self._handle_call_from_instr(func_name_instr, instr)