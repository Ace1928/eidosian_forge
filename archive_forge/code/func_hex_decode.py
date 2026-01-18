import codecs
import binascii
def hex_decode(input, errors='strict'):
    assert errors == 'strict'
    return (binascii.a2b_hex(input), len(input))