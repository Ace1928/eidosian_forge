from manilaclient.tests.functional.osc import base
def test_limits_show_rate(self):
    limits = self.listing_result('share', ' limits show --rate --print-empty')
    self.assertTableStruct(limits, ['Verb', 'Regex', 'URI', 'Value', 'Remaining', 'Unit', 'Next Available'])