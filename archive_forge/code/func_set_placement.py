from winappdbg import win32
def set_placement(self, placement):
    """
        Set the window placement in the desktop.

        @see: L{get_placement}

        @type  placement: L{win32.WindowPlacement}
        @param placement: Window placement in the desktop.

        @raise WindowsError: An error occured while processing this request.
        """
    win32.SetWindowPlacement(self.get_handle(), placement)