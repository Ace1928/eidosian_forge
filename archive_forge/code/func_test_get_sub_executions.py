from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
def test_get_sub_executions(self):
    url = self.TEST_URL + URL_TEMPLATE_SUB_EXECUTIONS % (EXEC['id'], '?max_depth=-1&errors_only=')
    self.requests_mock.get(url, json={'executions': [EXEC, SUB_WF_EXEC]})
    sub_execution_list = self.executions.get_ex_sub_executions(EXEC['id'])
    self.assertEqual(2, len(sub_execution_list))
    self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), sub_execution_list[0].to_dict())
    self.assertDictEqual(executions.Execution(self.executions, SUB_WF_EXEC).to_dict(), sub_execution_list[1].to_dict())