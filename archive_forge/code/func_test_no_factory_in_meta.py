import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
def test_no_factory_in_meta(self):
    _lb, flow_detail = p_utils.temporary_flow_detail()
    self.assertRaisesRegex(ValueError, '^Cannot .* no factory information saved.$', taskflow.engines.flow_from_detail, flow_detail)