@proxy_type.setter
def proxy_type(self, value) -> None:
    """Sets proxy type.

        :Args:
         - value: The proxy type.
        """
    self._verify_proxy_type_compatibility(value)
    self.proxyType = value