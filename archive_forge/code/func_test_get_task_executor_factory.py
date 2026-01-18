from unittest import mock
from glance.api import property_protections
from glance import context
from glance import gateway
from glance import notifier
from glance import quota
from glance.tests.unit import utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('glance.domain.TaskExecutorFactory')
def test_get_task_executor_factory(self, mock_factory):

    @mock.patch.object(self.gateway, 'get_task_repo')
    @mock.patch.object(self.gateway, 'get_repo')
    @mock.patch.object(self.gateway, 'get_image_factory')
    def _test(mock_gif, mock_gr, mock_gtr):
        self.gateway.get_task_executor_factory(self.context)
        mock_gtr.assert_called_once_with(self.context)
        mock_gr.assert_called_once_with(self.context)
        mock_gif.assert_called_once_with(self.context)
        mock_factory.assert_called_once_with(mock_gtr.return_value, mock_gr.return_value, mock_gif.return_value, admin_repo=None)
    _test()