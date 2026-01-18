from winappdbg import win32
def is_visible(self):
    """
        @see: {show}, {hide}
        @rtype:  bool
        @return: C{True} if the window is in a visible state.
        """
    return win32.IsWindowVisible(self.get_handle())