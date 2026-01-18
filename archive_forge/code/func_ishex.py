def ishex(c):
    """Return true if the byte ordinal 'c' is a hexadecimal digit in ASCII."""
    assert isinstance(c, bytes)
    return b'0' <= c <= b'9' or b'a' <= c <= b'f' or b'A' <= c <= b'F'