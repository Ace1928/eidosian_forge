import base64
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import nacl.public
import six
from macaroonbakery.bakery import _codec as codec
def test_empty_caveat_id(self):
    with self.assertRaises(bakery.VerificationError) as context:
        bakery.decode_caveat(self.tp_key, b'')
    self.assertTrue('empty third party caveat' in str(context.exception))