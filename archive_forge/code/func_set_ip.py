def set_ip(self, ip):
    """Will be used to set _ip point to current ipython instance b/f call

        Override this method if you don't want this to happen.

        """
    self._ip = ip