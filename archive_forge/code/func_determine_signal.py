import os
import signal
from typing import Optional
def determine_signal():
    global _breakin_signal_number
    global _breakin_signal_name
    if _breakin_signal_number is not None:
        return _breakin_signal_number
    sigquit = getattr(signal, 'SIGQUIT', None)
    sigbreak = getattr(signal, 'SIGBREAK', None)
    if sigquit is not None:
        _breakin_signal_number = sigquit
        _breakin_signal_name = 'SIGQUIT'
    elif sigbreak is not None:
        _breakin_signal_number = sigbreak
        _breakin_signal_name = 'SIGBREAK'
    return _breakin_signal_number