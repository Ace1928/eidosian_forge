import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
def test_no_arg_factory(self):
    name = 'some.test.factory'
    _lb, flow_detail = p_utils.temporary_flow_detail()
    flow_detail.meta = dict(factory=dict(name=name))
    with mock.patch('oslo_utils.importutils.import_class', return_value=lambda: 'RESULT') as mock_import:
        result = taskflow.engines.flow_from_detail(flow_detail)
        mock_import.assert_called_once_with(name)
    self.assertEqual('RESULT', result)