import codecs
def segregate(str):
    """3.1 Basic code point segregation"""
    base = bytearray()
    extended = set()
    for c in str:
        if ord(c) < 128:
            base.append(ord(c))
        else:
            extended.add(c)
    extended = sorted(extended)
    return (bytes(base), extended)