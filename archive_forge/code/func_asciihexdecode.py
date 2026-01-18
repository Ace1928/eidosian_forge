import re
import struct
def asciihexdecode(data: bytes) -> bytes:
    """
    ASCIIHexDecode filter: PDFReference v1.4 section 3.3.1
    For each pair of ASCII hexadecimal digits (0-9 and A-F or a-f), the
    ASCIIHexDecode filter produces one byte of binary data. All white-space
    characters are ignored. A right angle bracket character (>) indicates
    EOD. Any other characters will cause an error. If the filter encounters
    the EOD marker after reading an odd number of hexadecimal digits, it
    will behave as if a 0 followed the last digit.
    """

    def decode(x: bytes) -> bytes:
        i = int(x, 16)
        return bytes((i,))
    out = b''
    for x in hex_re.findall(data):
        out += decode(x)
    m = trail_re.search(data)
    if m:
        out += decode(m.group(1) + b'0')
    return out