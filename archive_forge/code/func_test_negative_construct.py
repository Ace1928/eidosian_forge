import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def test_negative_construct(self):
    coord = dict(point_x=10, point_y=4)
    coordG = dict(point_x=_curves['p521'].Gx, point_y=_curves['p521'].Gy)
    self.assertRaises(ValueError, ECC.construct, curve='P-521', **coord)
    self.assertRaises(ValueError, ECC.construct, curve='P-521', d=2, **coordG)