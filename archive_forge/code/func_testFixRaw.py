from srsly.msgpack import unpackb
def testFixRaw():
    check(b'\x94\xa0\xa1a\xa2bc\xa3def', (b'', b'a', b'bc', b'def'))