import _signal
from _signal import *
from enum import IntEnum as _IntEnum
@_wraps(_signal.sigwait)
def sigwait(sigset):
    retsig = _signal.sigwait(sigset)
    return _int_to_enum(retsig, Signals)