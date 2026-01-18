import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def getActiveWindowTitle():
    """Returns a string of the title text of the currently active (focused) Window."""
    global activeWindowTitle
    activeWindowHwnd = ctypes.windll.user32.GetForegroundWindow()
    if activeWindowHwnd == 0:
        return None

    def foreach_window(hWnd, lParam):
        global activeWindowTitle
        if hWnd == activeWindowHwnd:
            length = getWindowTextLength(hWnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            getWindowText(hWnd, buff, length + 1)
            activeWindowTitle = buff.value
        return True
    enumWindows(enumWindowsProc(foreach_window), 0)
    return activeWindowTitle