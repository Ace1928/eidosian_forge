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
def on_CALL_FUNCTION_VAR_KW(self, instr):
    _names_of_kw_args = self._stack.pop()
    arg = instr.arg
    argc = arg & 255
    argc += (arg >> 8) * 2
    self._stack.pop()
    for _ in range(argc):
        self._stack.pop()
    func_name_instr = self._stack.pop()
    self._handle_call_from_instr(func_name_instr, instr)