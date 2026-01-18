import uuid
from heat.common import short_id
from heat.tests import common
def test_get_id_uuid_f(self):
    source = uuid.UUID('ffffffff-ffff-4fff-8000-000000000000')
    self.assertEqual('777777777777', short_id.get_id(source))