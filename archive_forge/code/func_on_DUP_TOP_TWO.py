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
def on_DUP_TOP_TWO(self, instr):
    if len(self._stack) == 0:
        self._stack.append(instr)
        return
    if len(self._stack) == 1:
        i = self._stack[-1]
        self._stack.append(i)
        self._stack.append(instr)
        return
    i = self._stack[-1]
    j = self._stack[-2]
    self._stack.append(j)
    self._stack.append(i)