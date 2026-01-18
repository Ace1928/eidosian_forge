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
def on_CALL_FUNCTION(self, instr):
    arg = instr.arg
    argc = arg & 255
    argc += (arg >> 8) * 2
    for _ in range(argc):
        try:
            self._stack.pop()
        except IndexError:
            return
    try:
        func_name_instr = self._stack.pop()
    except IndexError:
        return
    self._handle_call_from_instr(func_name_instr, instr)