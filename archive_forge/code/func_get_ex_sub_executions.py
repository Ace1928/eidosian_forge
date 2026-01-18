from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from mistralclient.api import base
def get_ex_sub_executions(self, id, errors_only='', max_depth=-1):
    ex_sub_execs_path = '/executions/%s/executions%s'
    params = '?max_depth=%s&errors_only=%s' % (max_depth, errors_only)
    return self._list(ex_sub_execs_path % (id, params), response_key='executions')