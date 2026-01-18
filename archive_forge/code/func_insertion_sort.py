import codecs
def insertion_sort(base, extended, errors):
    """3.2 Insertion unsort coding"""
    char = 128
    pos = -1
    bias = 72
    extpos = 0
    while extpos < len(extended):
        newpos, delta = decode_generalized_number(extended, extpos, bias, errors)
        if delta is None:
            return base
        pos += delta + 1
        char += pos // (len(base) + 1)
        if char > 1114111:
            if errors == 'strict':
                raise UnicodeError('Invalid character U+%x' % char)
            char = ord('?')
        pos = pos % (len(base) + 1)
        base = base[:pos] + chr(char) + base[pos:]
        bias = adapt(delta, extpos == 0, len(base))
        extpos = newpos
    return base