from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def client_to_screen(self, hWnd):
    """
        Translates window client coordinates to screen coordinates.

        @see: L{screen_to_client}, L{translate}

        @type  hWnd: int or L{HWND} or L{system.Window}
        @param hWnd: Window handle.

        @rtype:  L{Rect}
        @return: New object containing the translated coordinates.
        """
    topleft = ClientToScreen(hWnd, (self.left, self.top))
    bottomright = ClientToScreen(hWnd, (self.bottom, self.right))
    return Rect(topleft.x, topleft.y, bottomright.x, bottomright.y)