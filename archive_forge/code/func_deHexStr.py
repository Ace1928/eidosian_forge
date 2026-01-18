import ast
import string
def deHexStr(hexdata):
    """Convert a hex string to binary data."""
    hexdata = strjoin(hexdata.split())
    if len(hexdata) % 2:
        hexdata = hexdata + '0'
    data = []
    for i in range(0, len(hexdata), 2):
        data.append(bytechr(int(hexdata[i:i + 2], 16)))
    return bytesjoin(data)