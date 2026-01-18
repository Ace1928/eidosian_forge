import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def assertUnhexlify(self, as_hex):
    ba_unhex = binascii.unhexlify(as_hex)
    mod_unhex = self.module._py_unhexlify(as_hex)
    if ba_unhex != mod_unhex:
        if mod_unhex is None:
            mod_hex = b'<None>'
        else:
            mod_hex = binascii.hexlify(mod_unhex)
        self.fail('_py_unhexlify returned a different answer from binascii:\n    %r\n != %r' % (binascii.hexlify(ba_unhex), mod_hex))