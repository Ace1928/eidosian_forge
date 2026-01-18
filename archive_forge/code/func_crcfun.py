import struct
import sys
def crcfun(data, crc=initCrc, table=_table, fun=_fun):
    return xorOut ^ fun(data, xorOut ^ crc, table)