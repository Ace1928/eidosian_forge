from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def axfr(self, zone):
    zone = v2_utils.resolve_by_name(self.list, zone)
    url = f'/zones/{zone}/tasks/xfr'
    self.client.session.post(url)