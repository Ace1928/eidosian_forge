import ctypes
from ctypes import wintypes # We can't use ctypes.wintypes, we must import wintypes this way.
from pygetwindow import PyGetWindowException, pointInRect, BaseWindow, Rect, Point, Size
def getWindowsWithTitle(title):
    """Returns a list of Window objects that substring match ``title`` in their title text."""
    hWndsAndTitles = _getAllTitles()
    windowObjs = []
    for hWnd, winTitle in hWndsAndTitles:
        if title.upper() in winTitle.upper():
            windowObjs.append(Win32Window(hWnd))
    return windowObjs