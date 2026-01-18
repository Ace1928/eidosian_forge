from unittest import mock
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.engine import worker
from heat.objects import stack as stack_objects
from heat.rpc import worker_client as wc
from heat.tests import common
from heat.tests import utils
@mock.patch.object(check_resource, 'load_resource')
@mock.patch.object(check_resource.CheckResource, 'check')
def test_check_resource_adds_and_removes_msg_queue(self, mock_check, mock_load_resource):
    mock_tgm = mock.MagicMock()
    mock_tgm.add_msg_queue = mock.Mock(return_value=None)
    mock_tgm.remove_msg_queue = mock.Mock(return_value=None)
    self.worker = worker.WorkerService('host-1', 'topic-1', 'engine_id', mock_tgm)
    ctx = utils.dummy_context()
    current_traversal = 'something'
    fake_res = mock.MagicMock()
    fake_res.current_traversal = current_traversal
    mock_load_resource.return_value = (fake_res, fake_res, fake_res)
    self.worker.check_resource(ctx, mock.Mock(), current_traversal, {}, mock.Mock(), mock.Mock())
    self.assertTrue(mock_tgm.add_msg_queue.called)
    self.assertTrue(mock_tgm.remove_msg_queue.called)