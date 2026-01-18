from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_os_volume_transfer_5678_accept(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    transfer1 = '5678'
    return (200, {}, {'transfer': _stub_transfer(transfer1, base_uri, tenant_id)})