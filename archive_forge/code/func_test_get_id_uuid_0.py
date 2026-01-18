import uuid
from heat.common import short_id
from heat.tests import common
def test_get_id_uuid_0(self):
    source = uuid.UUID('00000000-0000-4000-bfff-ffffffffffff')
    self.assertEqual('aaaaaaaaaaaa', short_id.get_id(source))