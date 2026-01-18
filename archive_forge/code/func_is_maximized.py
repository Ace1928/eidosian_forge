from winappdbg import win32
def is_maximized(self):
    """
        @see: L{maximize}
        @rtype:  bool
        @return: C{True} if the window is maximized.
        """
    return win32.IsZoomed(self.get_handle())