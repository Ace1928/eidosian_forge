from cinderclient.tests.functional import base
def test_absolute_limits(self):
    limits = self.cinder('absolute-limits')
    self.assertTableHeaders(limits, ['Name', 'Value'])