import os
import subprocess
import sys
from .error import TryNext
def clipboard_get(self):
    """ Get text from the clipboard.
    """
    from ..lib.clipboard import osx_clipboard_get, tkinter_clipboard_get, win32_clipboard_get, wayland_clipboard_get
    if sys.platform == 'win32':
        chain = [win32_clipboard_get, tkinter_clipboard_get]
    elif sys.platform == 'darwin':
        chain = [osx_clipboard_get, tkinter_clipboard_get]
    else:
        chain = [wayland_clipboard_get, tkinter_clipboard_get]
    dispatcher = CommandChainDispatcher()
    for func in chain:
        dispatcher.add(func)
    text = dispatcher()
    return text