import _signal
from _signal import *
from enum import IntEnum as _IntEnum
@_wraps(_signal.pthread_sigmask)
def pthread_sigmask(how, mask):
    sigs_set = _signal.pthread_sigmask(how, mask)
    return set((_int_to_enum(x, Signals) for x in sigs_set))