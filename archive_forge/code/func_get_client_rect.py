from winappdbg import win32
def get_client_rect(self):
    """
        Get the window's client area coordinates in the desktop.

        @rtype:  L{win32.Rect}
        @return: Rectangle occupied by the window's client area in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
    cr = win32.GetClientRect(self.get_handle())
    cr.left, cr.top = self.client_to_screen(cr.left, cr.top)
    cr.right, cr.bottom = self.client_to_screen(cr.right, cr.bottom)
    return cr