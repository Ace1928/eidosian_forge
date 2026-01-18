from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
def test_invalid_durations(self):
    self.assertRaises(ValueError, util.parse_isoduration, 'P1Y')
    self.assertRaises(ValueError, util.parse_isoduration, 'P1DT12H')
    self.assertRaises(ValueError, util.parse_isoduration, 'PT1Y1D')
    self.assertRaises(ValueError, util.parse_isoduration, 'PTAH1M0S')
    self.assertRaises(ValueError, util.parse_isoduration, 'PT1HBM0S')
    self.assertRaises(ValueError, util.parse_isoduration, 'PT1H1MCS')
    self.assertRaises(ValueError, util.parse_isoduration, 'PT1H1H')
    self.assertRaises(ValueError, util.parse_isoduration, 'PT1MM')
    self.assertRaises(ValueError, util.parse_isoduration, 'PT1S0S')
    self.assertRaises(ValueError, util.parse_isoduration, 'ABCDEFGH')