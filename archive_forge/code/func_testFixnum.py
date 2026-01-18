from srsly.msgpack import unpackb
def testFixnum():
    check(b'\x92\x93\x00@\x7f\x93\xe0\xf0\xff', ((0, 64, 127), (-32, -16, -1)))