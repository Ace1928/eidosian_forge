from heat.common import identifier
from heat.tests import common
def test_arn_url_decode_escape_round_trip(self):
    enc_arn = ''.join(['arn%3Aopenstack%3Aheat%3A%3A%253A%252F%3A', 'stacks%2F%253A%2525%2F%253A%252F%2F%253A'])
    url = self.url_prefix + enc_arn
    hi = identifier.HeatIdentifier.from_arn_url(url)
    hi2 = identifier.HeatIdentifier.from_arn_url(self.url_prefix + hi.arn_url_path())
    self.assertEqual(hi, hi2)