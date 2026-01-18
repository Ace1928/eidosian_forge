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
def get_smart_step_into_variant_from_frame_offset(frame_f_lasti, variants):
    """
    Given the frame.f_lasti, return the related `Variant`.

    :note: if the offset is found before any variant available or no variants are
           available, None is returned.

    :rtype: Variant|NoneType
    """
    if not variants:
        return None
    i = bisect(KeyifyList(variants, lambda entry: entry.offset), frame_f_lasti)
    if i == 0:
        return None
    else:
        return variants[i - 1]