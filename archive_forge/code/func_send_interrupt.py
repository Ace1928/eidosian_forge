import ctypes
from typing import Any
def send_interrupt(interrupt_handle: Any) -> None:
    """Sends an interrupt event using the specified handle."""
    ctypes.windll.kernel32.SetEvent(interrupt_handle)