from novaclient.tests.functional import base
def test_aggregate_update_az(self):
    self.nova('aggregate-create', params=self.agg2)
    self.nova('aggregate-update', params='--availability-zone=myaz %s' % self.agg2)
    output = self.nova('aggregate-show', params=self.agg2)
    self.assertIn('myaz', output)
    self.nova('aggregate-delete', params=self.agg2)