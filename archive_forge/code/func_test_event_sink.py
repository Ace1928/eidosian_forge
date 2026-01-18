from unittest import mock
from heat.engine.clients.os import zaqar
from heat.tests import common
from heat.tests import utils
def test_event_sink(self):
    context = utils.dummy_context()
    client = context.clients.client('zaqar')
    fake_queue = mock.MagicMock()
    client.queue = lambda x, auto_create: fake_queue
    sink = zaqar.ZaqarEventSink('myqueue')
    sink.consume(context, {'hello': 'world'})
    fake_queue.post.assert_called_once_with({'body': {'hello': 'world'}, 'ttl': 3600})