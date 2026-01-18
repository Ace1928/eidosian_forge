import enum
import threading
from cupyx.distributed import _klv_utils
def klv(self):
    v = bytearray(bytes(True))
    action = _klv_utils.get_result_action_t(0, v)
    return bytes(action)