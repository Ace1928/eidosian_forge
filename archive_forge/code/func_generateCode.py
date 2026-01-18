import struct
import sys
def generateCode(self, functionName, out, dataType=None, crcType=None):
    """Generate a C/C++ function.

        functionName -- String specifying the name of the function.

        out -- An open file-like object with a write method.  This specifies
        where the generated code is written.

        dataType -- An optional parameter specifying the data type of the input
        data to the function.  Defaults to UINT8.

        crcType -- An optional parameter specifying the data type of the CRC
        value.  Defaults to one of UINT8, UINT16, UINT32, or UINT64 depending
        on the size of the CRC value.
        """
    if dataType is None:
        dataType = 'UINT8'
    if crcType is None:
        size = 8 * self.digest_size
        if size == 24:
            size = 32
        crcType = 'UINT%d' % size
    if self.digest_size == 1:
        crcAlgor = 'table[*data ^ (%s)crc]'
    elif self.reverse:
        crcAlgor = 'table[*data ^ (%s)crc] ^ (crc >> 8)'
    else:
        shift = 8 * (self.digest_size - 1)
        crcAlgor = 'table[*data ^ (%%s)(crc >> %d)] ^ (crc << 8)' % shift
    fmt = '0x%%0%dX' % (2 * self.digest_size)
    if self.digest_size <= 4:
        fmt = fmt + 'U,'
    else:
        fmt = fmt + 'ULL,'
    n = {1: 8, 2: 8, 3: 4, 4: 4, 8: 2}[self.digest_size]
    lst = []
    for i, val in enumerate(self.table):
        if i % n == 0:
            lst.append('\n    ')
        lst.append(fmt % val)
    poly = 'polynomial: 0x%X' % self.poly
    if self.reverse:
        poly = poly + ', bit reverse algorithm'
    if self.xorOut:
        preCondition = '\n    crc = crc ^ %s;' % (fmt[:-1] % self.xorOut)
        postCondition = preCondition
    else:
        preCondition = ''
        postCondition = ''
    if self.digest_size == 3:
        if self.reverse:
            preCondition += '\n    crc = crc & 0xFFFFFFU;'
        else:
            postCondition += '\n    crc = crc & 0xFFFFFFU;'
    parms = {'dataType': dataType, 'crcType': crcType, 'name': functionName, 'crcAlgor': crcAlgor % dataType, 'crcTable': ''.join(lst), 'poly': poly, 'preCondition': preCondition, 'postCondition': postCondition}
    out.write(_codeTemplate % parms)