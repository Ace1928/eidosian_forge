import uuid
from heat.common import short_id
from heat.tests import common
def test_get_id_string(self):
    id = short_id.get_id('11111111-1111-4111-bfff-ffffffffffff')
    self.assertEqual('ceirceirceir', id)