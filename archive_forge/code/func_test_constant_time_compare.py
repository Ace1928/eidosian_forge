import hashlib
import hmac
from oslotest import base as test_base
import testscenarios
from oslo_utils import secretutils
def test_constant_time_compare(self):
    ctc = secretutils._constant_time_compare
    self.assertTrue(ctc(self.converter('abcd'), self.converter('abcd')))
    self.assertTrue(ctc(self.converter(''), self.converter('')))
    self.assertTrue(ctc('abcd', 'abcd'))
    self.assertFalse(ctc(self.converter('abcd'), self.converter('efgh')))
    self.assertFalse(ctc(self.converter('abc'), self.converter('abcd')))
    self.assertFalse(ctc(self.converter('abc'), self.converter('abc\x00')))
    self.assertFalse(ctc(self.converter(''), self.converter('abc')))
    self.assertTrue(ctc(self.converter('abcd1234'), self.converter('abcd1234')))
    self.assertFalse(ctc(self.converter('abcd1234'), self.converter('ABCD234')))
    self.assertFalse(ctc(self.converter('abcd1234'), self.converter('a')))
    self.assertFalse(ctc(self.converter('abcd1234'), self.converter('1234abcd')))
    self.assertFalse(ctc('abcd1234', '1234abcd'))