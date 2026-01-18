from oslo_serialization import jsonutils
from mistralclient.api import base
from mistralclient.api.v2 import executions
def get_task_sub_executions(self, id, errors_only='', max_depth=-1):
    task_sub_execs_path = '/tasks/%s/executions%s'
    params = '?max_depth=%s&errors_only=%s' % (max_depth, errors_only)
    return self._list(task_sub_execs_path % (id, params), response_key='executions', returned_res_cls=executions.Execution)