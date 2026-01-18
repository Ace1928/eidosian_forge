import ast
import string
def num2binary(l, bits=32):
    items = []
    binary = ''
    for i in range(bits):
        if l & 1:
            binary = '1' + binary
        else:
            binary = '0' + binary
        l = l >> 1
        if not (i + 1) % 8:
            items.append(binary)
            binary = ''
    if binary:
        items.append(binary)
    items.reverse()
    assert l in (0, -1), "number doesn't fit in number of bits"
    return ' '.join(items)