import uuid
from osc_placement.tests.functional import base
def test_granular_traits1(self):
    groups = {'1': {'resources': ('VCPU=6',)}, '2': {'resources': ('VCPU=10',), 'required': ['HW_CPU_X86_AVX']}}
    rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate')
    self.assertEqual(0, len(rows))