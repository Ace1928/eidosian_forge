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
def on_MAKE_FUNCTION(self, instr):
    if not IS_PY311_OR_GREATER:
        qualname = self._stack.pop()
        code_obj_instr = self._stack.pop()
    else:
        qualname = code_obj_instr = self._stack.pop()
    arg = instr.arg
    if arg & 8:
        _func_closure = self._stack.pop()
    if arg & 4:
        _func_annotations = self._stack.pop()
    if arg & 2:
        _func_kwdefaults = self._stack.pop()
    if arg & 1:
        _func_defaults = self._stack.pop()
    call_name = self._getcallname(qualname)
    if call_name in ('<genexpr>', '<listcomp>', '<setcomp>', '<dictcomp>'):
        if isinstance(code_obj_instr.arg, CodeType):
            self.func_name_id_to_code_object[_TargetIdHashable(qualname)] = code_obj_instr.arg
    self._stack.append(qualname)