import reportlab
def escapePDF(s):
    r = []
    for c in s:
        if not type(c) is int:
            c = ord(c)
        r.append(_ESCAPEDICT[c])
    return ''.join(r)