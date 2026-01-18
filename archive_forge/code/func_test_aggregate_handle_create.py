from unittest import mock
from heat.engine.clients.os import nova
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_aggregate_handle_create(self):
    value = mock.MagicMock()
    aggregate_id = '927202df-1afb-497f-8368-9c2d2f26e5db'
    value.id = aggregate_id
    self.aggregates.create.return_value = value
    self.my_aggregate.handle_create()
    value.set_metadata.assert_called_once_with({'availability_zone': 'nova'})
    self.assertEqual(2, value.add_host.call_count)
    self.assertEqual(aggregate_id, self.my_aggregate.resource_id)