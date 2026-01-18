import re
def normalise_unreserved(string):
    """Return a version of 's' where no unreserved characters are encoded.

    Unreserved characters are defined in Section 2.3 of RFC 3986.

    Percent encoded sequences are normalised to upper case.
    """
    result = string.split('%')
    unreserved = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
    for index, item in enumerate(result):
        if index == 0:
            continue
        try:
            ch = int(item[:2], 16)
        except ValueError:
            continue
        if chr(ch) in unreserved:
            result[index] = chr(ch) + item[2:]
        else:
            result[index] = '%%%02X%s' % (ch, item[2:])
    return ''.join(result)