from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def screen_to_client(self, hWnd):
    """
        Translates window screen coordinates to client coordinates.

        @see: L{client_to_screen}, L{translate}

        @type  hWnd: int or L{HWND} or L{system.Window}
        @param hWnd: Window handle.

        @rtype:  L{Rect}
        @return: New object containing the translated coordinates.
        """
    topleft = ScreenToClient(hWnd, (self.left, self.top))
    bottomright = ScreenToClient(hWnd, (self.bottom, self.right))
    return Rect(topleft.x, topleft.y, bottomright.x, bottomright.y)