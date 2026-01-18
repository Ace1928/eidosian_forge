from winappdbg import win32
def get_screen_rect(self):
    """
        Get the window coordinates in the desktop.

        @rtype:  L{win32.Rect}
        @return: Rectangle occupied by the window in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
    return win32.GetWindowRect(self.get_handle())