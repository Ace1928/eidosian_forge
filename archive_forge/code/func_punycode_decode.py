import codecs
def punycode_decode(text, errors):
    if isinstance(text, str):
        text = text.encode('ascii')
    if isinstance(text, memoryview):
        text = bytes(text)
    pos = text.rfind(b'-')
    if pos == -1:
        base = ''
        extended = str(text, 'ascii').upper()
    else:
        base = str(text[:pos], 'ascii', errors)
        extended = str(text[pos + 1:], 'ascii').upper()
    return insertion_sort(base, extended, errors)