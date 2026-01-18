import enum
import threading
from cupyx.distributed import _klv_utils
@staticmethod
def from_klv(klv):
    return True