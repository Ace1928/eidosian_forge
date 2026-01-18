from heat.common import identifier
from heat.tests import common
def test_arn_url_parse_round_trip(self):
    arn = '/arn%3Aopenstack%3Aheat%3A%3At%3Astacks/s/i/p'
    url = 'http://1.2.3.4/foo' + arn
    hi = identifier.HeatIdentifier.from_arn_url(url)
    self.assertEqual(arn, hi.arn_url_path())