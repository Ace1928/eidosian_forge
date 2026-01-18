import hashlib
import json
from oslo_utils import encodeutils
from requests import codes
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import schemas
@utils.add_req_id_to_object()
def get_stores_info_detail(self):
    """Get available stores info from discovery endpoint."""
    url = '/v2/info/stores/detail'
    resp, body = self.http_client.get(url)
    return (body, resp)