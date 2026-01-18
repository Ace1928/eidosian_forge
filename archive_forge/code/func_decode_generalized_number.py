import codecs
def decode_generalized_number(extended, extpos, bias, errors):
    """3.3 Generalized variable-length integers"""
    result = 0
    w = 1
    j = 0
    while 1:
        try:
            char = ord(extended[extpos])
        except IndexError:
            if errors == 'strict':
                raise UnicodeError('incomplete punicode string')
            return (extpos + 1, None)
        extpos += 1
        if 65 <= char <= 90:
            digit = char - 65
        elif 48 <= char <= 57:
            digit = char - 22
        elif errors == 'strict':
            raise UnicodeError("Invalid extended code point '%s'" % extended[extpos - 1])
        else:
            return (extpos, None)
        t = T(j, bias)
        result += digit * w
        if digit < t:
            return (extpos, result)
        w = w * (36 - t)
        j += 1