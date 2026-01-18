from winappdbg import win32
def maximize(self, bAsync=True):
    """
        Maximize the window.

        @see: L{minimize}, L{restore}

        @type  bAsync: bool
        @param bAsync: Perform the request asynchronously.

        @raise WindowsError: An error occured while processing this request.
        """
    if bAsync:
        win32.ShowWindowAsync(self.get_handle(), win32.SW_MAXIMIZE)
    else:
        win32.ShowWindow(self.get_handle(), win32.SW_MAXIMIZE)