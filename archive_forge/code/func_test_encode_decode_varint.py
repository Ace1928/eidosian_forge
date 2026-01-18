import base64
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import nacl.public
import six
from macaroonbakery.bakery import _codec as codec
def test_encode_decode_varint(self):
    tests = [(12, [12]), (127, [127]), (128, [128, 1]), (129, [129, 1]), (1234567, [135, 173, 75]), (12131231231312, [208, 218, 233, 173, 136, 225, 2])]
    for test in tests:
        data = bytearray()
        expected = bytearray()
        bakery.encode_uvarint(test[0], data)
        for v in test[1]:
            expected.append(v)
        self.assertEqual(data, expected)
        val = codec.decode_uvarint(bytes(data))
        self.assertEqual(test[0], val[0])
        self.assertEqual(len(test[1]), val[1])