from winappdbg import win32
def get_placement(self):
    """
        Retrieve the window placement in the desktop.

        @see: L{set_placement}

        @rtype:  L{win32.WindowPlacement}
        @return: Window placement in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
    return win32.GetWindowPlacement(self.get_handle())