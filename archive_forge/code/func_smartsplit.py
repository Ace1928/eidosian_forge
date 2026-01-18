import re
def smartsplit(code):
    """Split `code` at " symbol, only if it is not escaped."""
    strings = []
    pos = 0
    while pos < len(code):
        if code[pos] == '"':
            word = ''
            pos += 1
            while pos < len(code):
                if code[pos] == '"':
                    break
                if code[pos] == '\\':
                    word += '\\'
                    pos += 1
                word += code[pos]
                pos += 1
            strings.append('"%s"' % word)
        pos += 1
    return strings