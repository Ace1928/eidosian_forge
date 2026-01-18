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
def calculate_smart_step_into_variants(frame, start_line, end_line, base=0):
    """
    Calculate smart step into variants for the given line range.
    :param frame:
    :type frame: :py:class:`types.FrameType`
    :param start_line:
    :param end_line:
    :return: A list of call names from the first to the last.
    :note: it's guaranteed that the offsets appear in order.
    :raise: :py:class:`RuntimeError` if failed to parse the bytecode or if dis cannot be used.
    """
    variants = []
    code = frame.f_code
    lasti = frame.f_lasti
    call_order_cache = {}
    if DEBUG:
        print('dis.dis:')
        if IS_PY311_OR_GREATER:
            dis.dis(code, show_caches=False)
        else:
            dis.dis(code)
    for target in _get_smart_step_into_targets(code):
        variant = _convert_target_to_variant(target, start_line, end_line, call_order_cache, lasti, base)
        if variant is None:
            continue
        variants.append(variant)
    return variants