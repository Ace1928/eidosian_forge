from srsly.msgpack import unpackb
def testSimpleValue():
    check(b'\x93\xc0\xc2\xc3', (None, False, True))