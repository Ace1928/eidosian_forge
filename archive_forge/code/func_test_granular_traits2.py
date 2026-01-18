import uuid
from osc_placement.tests.functional import base
def test_granular_traits2(self):
    groups = {'1': {'resources': ('VCPU=6',)}, '2': {'resources': ('VCPU=10',), 'required': ['HW_CPU_X86_SSE']}}
    rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate')
    self.assertEqual(2, len(rows))
    numbers = {row['#'] for row in rows}
    self.assertEqual(1, len(numbers))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual(2, len(rps))
    self.assertIn(self.rp1_1['uuid'], rps)
    self.assertIn(self.rp1_2['uuid'], rps)