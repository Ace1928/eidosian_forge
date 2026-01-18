import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
def test_factory_with_arg(self):
    name = 'some.test.factory'
    _lb, flow_detail = p_utils.temporary_flow_detail()
    flow_detail.meta = dict(factory=dict(name=name, args=['foo']))
    with mock.patch('oslo_utils.importutils.import_class', return_value=lambda x: 'RESULT %s' % x) as mock_import:
        result = taskflow.engines.flow_from_detail(flow_detail)
        mock_import.assert_called_once_with(name)
    self.assertEqual('RESULT foo', result)