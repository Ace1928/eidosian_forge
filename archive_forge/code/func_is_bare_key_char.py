import string
def is_bare_key_char(self) -> bool:
    """
        Whether the character is a valid bare key name or not.
        """
    return self in self.BARE