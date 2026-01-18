import signal
from types import FrameType
from typing import Any, Callable, Dict, Optional, Union
def install_shutdown_handlers(function: SignalHandlerT, override_sigint: bool=True) -> None:
    """Install the given function as a signal handler for all common shutdown
    signals (such as SIGINT, SIGTERM, etc). If ``override_sigint`` is ``False`` the
    SIGINT handler won't be installed if there is already a handler in place
    (e.g. Pdb)
    """
    signal.signal(signal.SIGTERM, function)
    if signal.getsignal(signal.SIGINT) == signal.default_int_handler or override_sigint:
        signal.signal(signal.SIGINT, function)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, function)