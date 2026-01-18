from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def list_requests(self):
    url = '/zones/tasks/transfer_requests'
    return self._get(url, response_key='transfer_requests')