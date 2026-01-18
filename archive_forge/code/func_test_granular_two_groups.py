import uuid
from osc_placement.tests.functional import base
def test_granular_two_groups(self):
    groups = {'1': {'resources': ('VCPU=3',)}, '2': {'resources': ('VCPU=3',)}}
    rows = self.allocation_candidate_granular(groups=groups)
    self.assertEqual(6, len(rows))
    numbers = {row['#'] for row in rows}
    self.assertEqual(4, len(numbers))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual(2, len(rps))
    self.assertIn(self.rp1_1['uuid'], rps)
    self.assertIn(self.rp1_2['uuid'], rps)
    rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate')
    self.assertEqual(4, len(rows))
    numbers = {row['#'] for row in rows}
    self.assertEqual(2, len(numbers))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual(2, len(rps))
    self.assertIn(self.rp1_1['uuid'], rps)
    self.assertIn(self.rp1_2['uuid'], rps)
    rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate', limit=1)
    self.assertEqual(2, len(rows))
    numbers = {row['#'] for row in rows}
    self.assertEqual(1, len(numbers))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual(2, len(rps))
    self.assertIn(self.rp1_1['uuid'], rps)
    self.assertIn(self.rp1_2['uuid'], rps)