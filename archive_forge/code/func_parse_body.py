from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def parse_body(self):
    if not self.body:
        return None
    self.body = self.body.replace('\\.', '\\\\.')
    data = json.loads(self.body)
    return data