@header_encoding.setter
def header_encoding(self, value):
    """
        Enforces constraints on the value of header encoding.
        """
    if not isinstance(value, (bool, str, type(None))):
        raise ValueError('header_encoding must be bool, string, or None')
    if value is True:
        raise ValueError('header_encoding cannot be True')
    self._header_encoding = value