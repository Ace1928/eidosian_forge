def parse_number(s, start=0):
    """Parses a positive decimal integer number from the string.

    A tuple is returned (number, chars_consumed).  If the
    string is not a valid decimal number, then (None,0) is returned.
    """
    if start >= len(s):
        raise ParseError('Starting position is beyond the end of the string', s, start)
    if s[start] not in DIGIT:
        return (None, 0)
    pos = start
    n = 0
    while pos < len(s):
        c = s[pos]
        if c in DIGIT:
            n *= 10
            n += ord(c) - ord('0')
            pos += 1
        else:
            break
    return (n, pos - start)