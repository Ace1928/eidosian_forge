from heat.common import identifier
from heat.tests import common
def test_arn_round_trip(self):
    hii = identifier.HeatIdentifier('t', 's', 'i', 'p')
    hio = identifier.HeatIdentifier.from_arn(hii.arn())
    self.assertEqual(hii.tenant, hio.tenant)
    self.assertEqual(hii.stack_name, hio.stack_name)
    self.assertEqual(hii.stack_id, hio.stack_id)
    self.assertEqual(hii.path, hio.path)