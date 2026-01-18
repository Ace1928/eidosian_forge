from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def get_accept(self, accept_id):
    url = f'/zones/tasks/transfer_accepts/{accept_id}'
    return self._get(url)