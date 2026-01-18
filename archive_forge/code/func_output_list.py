from urllib import parse
from heatclient._i18n import _
from heatclient.common import base
from heatclient.common import utils
from heatclient import exc
def output_list(self, stack_id):
    stack_identifier = self._resolve_stack_id(stack_id)
    resp = self.client.get('/stacks/%s/outputs' % stack_identifier)
    body = utils.get_response_body(resp)
    return body