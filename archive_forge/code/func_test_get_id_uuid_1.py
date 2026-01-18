import uuid
from heat.common import short_id
from heat.tests import common
def test_get_id_uuid_1(self):
    source = uuid.UUID('11111111-1111-4111-bfff-ffffffffffff')
    self.assertEqual(76861433640456465, source.time)
    self.assertEqual('ceirceirceir', short_id.get_id(source))