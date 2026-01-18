from srsly.msgpack import unpackb
def testFixMap():
    check(b'\x82\xc2\x81\xc0\xc0\xc3\x81\xc0\x80', {False: {None: None}, True: {None: {}}})