def needsquoting(c, quotetabs, header):
    """Decide whether a particular byte ordinal needs to be quoted.

    The 'quotetabs' flag indicates whether embedded tabs and spaces should be
    quoted.  Note that line-ending tabs and spaces are always encoded, as per
    RFC 1521.
    """
    assert isinstance(c, bytes)
    if c in b' \t':
        return quotetabs
    if c == b'_':
        return header
    return c == ESCAPE or not b' ' <= c <= b'~'