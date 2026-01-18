import os
import signal
from typing import Optional
def hook_debugger_to_signal():
    """Add a signal handler so we drop into the debugger.

    On Unix, this is hooked into SIGQUIT (C-\\), and on Windows, this is
    hooked into SIGBREAK (C-Pause).
    """
    if os.environ.get('BRZ_SIGQUIT_PDB', '1') == '0':
        return
    sig = determine_signal()
    if sig is None:
        return
    signal.signal(sig, _debug)